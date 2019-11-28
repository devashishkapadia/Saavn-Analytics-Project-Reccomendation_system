package saavn_reccomandation_system;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.api.java.UDF1;

import scala.collection.Seq;

import java.io.Serializable;
import java.util.List;
public class udf implements Serializable {
	private static final long serialVersionUID = 1L;
	UDF1<Seq<Float>, Vector> toVector = new UDF1<Seq<Float>, Vector>(){
	public Vector call(Seq<Float> t1) throws Exception {

		    List<Float> L = scala.collection.JavaConversions.seqAsJavaList(t1);
		    double[] DoubleArray = new double[t1.length()]; 
		    for (int i = 0 ; i < L.size(); i++) { 
		      DoubleArray[i]=L.get(i); 
		    } 
		    return Vectors.dense(DoubleArray); 
		  } 
	};
}
