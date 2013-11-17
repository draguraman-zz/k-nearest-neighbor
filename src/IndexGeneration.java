import java.io.*; 
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel; 
import java.util.*;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


/// This application takes the raw inverted index generated from InvertedIndex 
/// and generates two index files: (1) term lexicon with basic term statistics and
/// pointers to the posting files. (2) postings.  
/// All the files are HDFS files, so they can potentially support indexing large collections
///
/// The program goes through the original raw inverted index generated from InvertedIndex sequentially
/// and rewrite the postings to a new file and record the starting points of entries for each term
/// in the term lexicon.
///
/// Note that the generated posting file is actually very similar to the original inverted index file,
/// so we could have kept the original inverted index file as the posting file, but in a polished
/// inverted index, we would represent everything as integers and compress them, so this extra 
/// step is conceptually necessary. 
///
/// usage: hadoop jar simir.jar IndexGeneration Path-to-rawPosting IndexFileName
/// "path-to-rawposting" points to a raw inverted index/posting file generated by InvertedIndex
/// (the current implementation can only take one file)
///  "IndexFileName" is the name (including the path) for the inverted index to be created.
/// Two files will be generated: "IndexFileName.lex" for the lexicon and "IndexFileName.pos" for postings.
 
public class IndexGeneration {

    public static void main (String [] args) throws IOException {

	/// the following is basic setup needed to access HDFS files
	Configuration conf = new Configuration();
	FileSystem fs = FileSystem.get(conf);
	FSDataInputStream fin;
	FSDataOutputStream foutposting, foutlexicon; 

	try { 
	    fin = fs.open(new Path(args[0] ));  // args[0] has the path to the raw inverted index
	    foutposting = fs.create(new Path(args[1] + ".pos" )); // posting file with name in args[1]
	    foutlexicon = fs.create(new Path(args[1] + ".lex")); // term lexicon with name in args[1]
	    char c;
	    String t=null;
	    int progress=0;
	    int freq;
	    String x="";
	    BufferedReader reader = new BufferedReader(new InputStreamReader(fin));
	    while ((t=reader.readLine()) != null) {
		//System.out.println(t);
		// each line corresponds to all the entries for a different term
		// it starts with the term itself with a sequence of (docID, termFreq) pairs 
		// representing the documents containing the term as well as the corresponding term counts. 
		StringTokenizer st = new StringTokenizer(t);
		x = st.nextToken();
		//System.out.println(x);
		foutlexicon.writeUTF(x);  // fetch the first string (which is the term) and
		// write it to the lexicon

		int df=0; 
		int count=0;
		long pos = foutposting.getPos(); 
		// remember the current position in the new posting file  
		// so that we can easily calculate the span of the postings for this term later

		while (st.hasMoreTokens()) {
		    // iterate over all the (docID count) pairs and copy them to foutposting. 
		    // first, copy the docID using foutposting.writeUTF. 
		    x=st.nextToken();
		    System.out.println(x);
		    foutposting.writeUTF(x);
		    
		    if (st.hasMoreTokens()) {
			// we should expect another token for the term frequency/count
			freq = Integer.parseInt(st.nextToken().trim()); 
			//#########################################################//
			// add a statement here so that in the end of the loop "count" would have the total 
			// count of the term in all the documents
			// Hint: how to update "count"? 
			// 
			//#########################################################//
			count = count + freq;
			foutposting.writeInt(freq); // copy the term frequency/count to foutposting
		    } else {
			System.err.println("Term frequency is expected");
		    }

		    //#########################################################//
		    // add a statement here to use "df" to count how many documents contain the term 
		    // Hint: how to update "df"? 
		    // 
		    //#########################################################//
		    df = df + 1;
		}
		int len= new Long(foutposting.getPos()-pos).intValue(); // this tells us the span of the postering entries for this term

		// the following four statements write out df, count, pos, and len to foutlexicon 
		// recall that the term was already written to foutlexicon. 
		foutlexicon.writeInt(df);
		foutlexicon.writeInt(count);
		foutlexicon.writeLong(pos);
		foutlexicon.writeInt(len);
		progress++;
		System.out.println(df + " " + count);
		if (progress % 5000 ==0) {
		    System.out.println(progress + " terms processed"); 
		}
	    }
	    foutlexicon.close();
	    foutposting.close(); 
	} catch (IOException ioe) {
	    System.out.println("can't open file "+args[0] + " or can't create the term index lexicon:"  + args[1]); 
            System.exit(1);
	}
	
	
    }
    
}