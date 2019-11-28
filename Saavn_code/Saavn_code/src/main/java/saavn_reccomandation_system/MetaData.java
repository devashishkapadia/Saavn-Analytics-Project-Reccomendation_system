package saavn_reccomandation_system;

import java.io.Serializable;

public class MetaData {
	public static class SongMetaData implements Serializable {
	    /**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		private String[] artistIds;
	    private String songId;

	    public String getSongId() {
	      return songId;
	    }   

	    public void setSongId(String sId) {
	      this.songId = sId;
	    }   

	    public String[] getArtistIds() {
	      return artistIds;
	    }   

	    public void setArtistIds(String[] aIds) {
	      this.artistIds = aIds;
	    }   
	}
}
