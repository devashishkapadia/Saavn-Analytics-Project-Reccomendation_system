package saavn_reccomandation_system;
import static org.apache.spark.sql.functions.col;


import java.util.Arrays;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import saavn_reccomandation_system.MetaData.SongMetaData;


public class project_code {
	
	
	public static SparkSession sparkSession;
	
    public static void main( String[] args ) throws AnalysisException
    {
		
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		
	if (args.length != 7) {
			System.out.println(
				"Usage: spark2-submit --master yarn --class saavn_reccomandation_system.project_code  SaavnAnalytics-0.0.1-SNAPSHOT.jar "
						+ "fs.s3.awsAccessKeyId"	 								//args[0]
						+ "fs.s3.awsSecretAccessKey"								//args[1]
					 	+ "s3a://bigdataanalyticsupgrad/activity/sample100mb.csv"   //args[2]
					 	+ "s3a://bigdataanalyticsupgrad/newmetadata/*"  			//args[3]
					 	+ "s3a://bigdataanalyticsupgrad/notification_actor/notification.csv" //args[4]
					 	+ "s3a://bigdataanalyticsupgrad/notification_clicks/*"  //args[5]
					 	+ "output_dir_path"); //arg[6]
			return;
		}
		
		// setting up connection with spark
		sparkSession = SparkSession.builder()
				.config("fs.s3.awsAccessKeyId", args[0])
				.config("fs.s3.awsSecretAccessKey", args[1])
				.config("spark.sql.broadcastTimeout", "36000")
				.appName("RecomendationSystem")
				.master("yarn")
				.getOrCreate();
			
		//Read user click stream data - path in args[0]			
		Dataset<Row> userProfile = 
				sparkSession.read().option("header", "false")
			   .csv(args[2])
				.toDF("UserId", "TimeStamp", "SongId", "Date");
		
		
		userProfile = userProfile.drop("TimeStamp", "Date");
		userProfile = userProfile.na().drop();

		//Get the song frequency per user by group by operation		
		Dataset<Row> userRatings = 
				userProfile.groupBy("UserId", "SongId")
				.count()
				.toDF("UserId", "SongId", "Frequency");

		//Create UserIndex(Integer Type) for string UserId column to use in ALS model
		StringIndexer indexer = new StringIndexer()
				.setInputCol("UserId")
				.setOutputCol("UserIndex");

		//Table columns - UserId, SongId, Frequency, UserIndex
		Dataset<Row> userIndexed = indexer.fit(userRatings).transform(userRatings);
			
		//Create SongIndex(Integer Type) for string SongId column to use in ALS model
		indexer.setInputCol("SongId").setOutputCol("SongIndex");

		//Table columns - UserId, SongId, Frequency, UserIndex
		Dataset<Row> songIndexed =
				indexer.fit(userIndexed).transform(userIndexed);
		
		//Cast UserIndex, SongIndex to Interger Type to use in ALS model
		// <UserId,UserIndex,SongId,SongIndex,Frequency>
		Dataset<Row> modelIndexed = songIndexed
				.withColumn("UserIndex", col("UserIndex").cast(DataTypes.IntegerType))
				.withColumn("SongIndex", col("SongIndex").cast(DataTypes.IntegerType));
	
		ALS als = new ALS()
				  .setRank(10)
				  .setMaxIter(5)
				  .setRegParam(0.01)
				  .setUserCol("UserIndex")
				  .setItemCol("SongIndex")
				  .setRatingCol("Frequency");
		ALSModel model = als.fit(modelIndexed);
		
		// Get the userFactors from ALS model to use it in kmeans
		Dataset<Row> userALSFeatures = model.userFactors();

		// <UserId,UserIndex>
		Dataset<Row> userIdTable = modelIndexed
										.drop("SongIndex","SongId","Frequency")
										.groupBy("UserId","UserIndex").count().drop("count");
		// <UserId,UserIndex,features(array)>
				Dataset<Row> userTableInfo = 
						userIdTable.join(userALSFeatures, userIdTable.col("UserIndex").equalTo(userALSFeatures.col("id"))).drop("id");
    		
			udf uf = new udf();
			sparkSession.udf().register("toVector", uf.toVector, new VectorUDT());
			Dataset<Row> userAlsFeatureVect = 
					userTableInfo.withColumn("featuresVect", functions.callUDF("toVector", userTableInfo.col("features"))).drop("features");
			// <UserId,UserIndex,alsfeatures(vector)>
			userAlsFeatureVect = userAlsFeatureVect.toDF("UserId", "UserIndex", "alsmodelfeatures");
		
			//Scale the alsfeatures before giving to kmeans
    		StandardScaler scaler = new StandardScaler()
    		  .setInputCol("alsmodelfeatures")
    		  .setOutputCol("scaledFseatures")
    		  .setWithStd(true)
    		  .setWithMean(true);

    		// Compute summary statistics by fitting the StandardScaler
    		StandardScalerModel scalerModel = scaler.fit(userAlsFeatureVect);

    		// Normalize each feature to have unit standard deviation.
    		Dataset<Row> scaledData = scalerModel.transform(userAlsFeatureVect);
    		
    		// <UserId,UserIndex, features(vector)>
    		scaledData = scaledData.drop("alsmodelfeatures").toDF("UserId", "UserIndex", "features");
	
			// Trains a k-means model, given array of k's and scaled and non scaled data
		//List<Integer> numClusters = Arrays.asList(180,200,230,240,260,280,300);
	   	/*List<Integer> numClusters = Arrays.asList(7);
		for (Integer k : numClusters) {
			KMeans kmeans = new KMeans().setK(k).setSeed(1L);
			KMeansModel modelk = kmeans.fit(scaledData);
		
			//Within Set Sum of Square Error (WESSE).					
			double WSSSE = modelk.computeCost(scaledData);
			System.out.println("WSSSE = " + WSSSE);
			
			//s the results
			
			Vector[] centers = modelk.clusterCenters();
			System.out.println("Cluster Centers for k: " + k + " ");
			for (Vector center: centers) {
			  System.out.println(center);
			}
			
		}*/
			 //Training kmeans with the results from above task. 
		KMeans kmeansFinal = new KMeans().setK(240).setSeed(1L);
		KMeansModel modelFinal = kmeansFinal.fit(scaledData);

		// Make Predictions for scaled user ratings data
		Dataset<Row> usersClusterInfo = modelFinal.transform(scaledData);
		
		
		userProfile = userProfile.toDF("UId", "song_id");

		// <song_id,UserId,prediction>
		Dataset<Row> userProfilePrediction = 
				userProfile.join(usersClusterInfo, userProfile.col("UId").equalTo(usersClusterInfo.col("UserId")))
				.drop("features","UId","UserIndex");
		

		// Read the metadata to get song to artistid mapping - path in args[1]
		String songMetaDataPath = args[3];
        JavaRDD<SongMetaData> songMetaRDD = sparkSession.read().textFile(songMetaDataPath).javaRDD()
				.map(line -> {
					String[] data1 = line.split(",");
					SongMetaData sm = new SongMetaData();
					sm.setSongId(data1[0]);
					sm.setArtistIds(Arrays.copyOfRange(data1, 1, data1.length));
					return sm;
				});
		
        Dataset<Row> songMetaDF = sparkSession.createDataFrame(songMetaRDD, SongMetaData.class);
        songMetaDF = songMetaDF.na().drop();
        
        
 
		// <UserId,prediction(cluserid),songId,artistIdss(array)>
		Dataset<Row> userClusterJoinSongArtistInfo =
				userProfilePrediction.join(songMetaDF, userProfilePrediction.col("song_id")
						.equalTo(songMetaDF.col("songId"))).drop("song_id");
		
		userClusterJoinSongArtistInfo =
				userClusterJoinSongArtistInfo.withColumn("artistIds", functions.explode(userClusterJoinSongArtistInfo.col("artistIds")));
		// <UserId,prediction(cluserid),songId,artistIdss>	
		
			
		
		Dataset<Row> popularArtistPerCluster = 
				userClusterJoinSongArtistInfo.groupBy("prediction", "artistIds")
				.count()
				.toDF("ClusterId", "ArtistId", "Frequency");
	
		popularArtistPerCluster.createTempView("ClusterArtistFreq");
		
		// <CluserId,ArtistId,Frequency,rank>
		Dataset<Row> rankArtistPerCluster = 
				sparkSession.sql("SELECT ClusterId,ArtistId,Frequency, rank from "
						+ "(SELECT ClusterId,ArtistId,Frequency, row_number() over(partition by ClusterId order by Frequency desc) as rank"
						+ " from ClusterArtistFreq) a WHERE rank == 1 order by a.Frequency desc");
		
		// Remove duplicate ArtistId assigned to multiple cluster - 1 Artistid = 1 cluser_id
		popularArtistPerCluster = rankArtistPerCluster.dropDuplicates("ArtistId");
		
		
		// Notification data <notifyId,ArtistId> input
		String notificationPath = args[4];
		Dataset<Row> notifyData =
				sparkSession.read().format("csv").
				option("header","false").load(notificationPath).
				toDF("notifyId", "Artist_Id");
		
		// Cleansing the notification data
		notifyData = notifyData.na().drop();
		
		// Get unique column of valid notifyId
		Dataset<Row> validNotifyId = notifyData.drop("Artist_Id").distinct();
		
		// Join notify data to poperArtistCluster table to get notifyId,clusterId,ArtistId table
		notifyData = 
				notifyData.join(popularArtistPerCluster,
						notifyData.col("Artist_Id").equalTo(popularArtistPerCluster.col("ArtistId")),
						"left_outer").drop("Artist_Id","Frequency","rank");
		
		
		// <notifyId, ClusterId>
		Dataset<Row> notifyIdClusterMap = 
				notifyData.groupBy("notifyId","ClusterId").count().drop("count");
	
		// getting UserClusterArtist <ClusterId,ArtistId,UserId>
		Dataset<Row> userclusterinfo =
				popularArtistPerCluster.join(userClusterJoinSongArtistInfo,
		popularArtistPerCluster.col("ClusterId").equalTo(userClusterJoinSongArtistInfo.col("prediction")),"left_outer")
				.drop("Frequency","rank","prediction","artistIds","songId");
		
		userclusterinfo = userclusterinfo.distinct();
		
		userclusterinfo.repartition(1).write().option("Header", "True").csv(args[6] + "/UserClusterArtist");
		Dataset<Row> notifyClusterUserArtistInfo =
				notifyIdClusterMap.join(userClusterJoinSongArtistInfo,
						notifyIdClusterMap.col("ClusterId").equalTo(userClusterJoinSongArtistInfo.col("prediction")),"left_outer");
		

		notifyClusterUserArtistInfo = notifyClusterUserArtistInfo.drop("prediction","songId");
		
		// <UserId, prediction>
		Dataset<Row> clusterUserMap = usersClusterInfo.drop("UserIndex","features");
	
		// <notifyId,ClusterId,UserId>
		Dataset<Row> notifyCluserUserMap =
				notifyIdClusterMap.join(clusterUserMap, notifyIdClusterMap.col("ClusterId")
						.equalTo(clusterUserMap.col("prediction")), "left_outer").drop("prediction");
		
		Dataset<Row> notifyClusterUserSendCount = notifyCluserUserMap.groupBy("notifyId").count();
		notifyClusterUserSendCount = notifyClusterUserSendCount.toDF("notifyId","UserSendCount");

		
		// Notification Clicks data input - path in args[3]
        String Notification_clicks_path = args[5];
		Dataset<Row> notify_clicks =
				sparkSession.read().format("csv").option("header","false").load(Notification_clicks_path).toDF("notify_Id","UserId","Date");
		
		// Cleansing - Removing invalid notification id rows - <notifyId, UserId>
		notify_clicks = 
				notify_clicks.join(validNotifyId, notify_clicks.col("notify_Id").equalTo(validNotifyId.col("notifyId")),"left_outer");
		
		// <notifyId,UserId>
		notify_clicks = notify_clicks.na().drop().drop("notifyId","Date").toDF("notify_Id","user_Id");

		Dataset<Row> notifyMatchingUserClicks = 
				notify_clicks.join(notifyCluserUserMap, notify_clicks.col("notify_Id")
						.equalTo(notifyCluserUserMap.col("notifyId"))
						.and(notify_clicks.col("user_Id").equalTo(notifyCluserUserMap.col("UserId"))), "left_outer");
	//droping all the the null records in notifyMatchingUserClicks
		notifyMatchingUserClicks = notifyMatchingUserClicks.na().drop();
		//taking the count for calculating how many user have been pushed the notification
		notifyMatchingUserClicks = notifyMatchingUserClicks.groupBy("notifyId").count();
		//creating column of click count for the calculation of ctr.
		notifyMatchingUserClicks = notifyMatchingUserClicks.toDF("notify_Id","click_cnt");
		
		Dataset<Row> notifyCTR =  
				notifyClusterUserSendCount.join(notifyMatchingUserClicks, notifyClusterUserSendCount.col("notifyId")
						.equalTo(notifyMatchingUserClicks.col("notify_Id")),"left_outer").drop("notify_Id");
        
		notifyCTR = notifyCTR.withColumn("CTR", notifyCTR.col("click_cnt").divide(notifyCTR.col("UserSendCount")));
		notifyCTR.createTempView("notifyCTR");
		// taking top 5 ctrs from the table .
		Dataset<Row> ctr= sparkSession.sql("select notifyId,UserSendCount,click_cnt ,CTR from notifyCTR order by CTR desc limit 5 ");
		System.out.println("-------------------------------Saving result CTR--------------------------------------");
		//using coalesce to create only one partition of the result .
		ctr.coalesce(1).write().option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
		.option("header","true") //Write the headercsv("data/CTR");
		.csv(args[6] + "/CTR");
	     ctr =ctr.withColumnRenamed("notifyId", "nfId");
	// performing broadcast join from ctr to notifyClusterUserArtistInfo to get only top 5 notification <Userid, artistid>
		Dataset<Row> NotificationNumber = ctr.join(notifyClusterUserArtistInfo,(ctr.col("nfId")).equalTo(notifyClusterUserArtistInfo.col("notifyId")))
				.drop("UserSendCount","nfId","click_cnt","CTR","ClusterId");
		NotificationNumber.show();
		System.out.println("-------------------------------Saving result NotificationNumber--------------------------------------");
		NotificationNumber.repartition(NotificationNumber.col("notifyId"))
			.write().option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false"). //Avoid creating of crc files
			option("header","true").partitionBy("notifyId").mode(SaveMode.Overwrite).csv(args[6]+ "/NotificationNumber");
		  sparkSession.stop();
		
    }
	

}
