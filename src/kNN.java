import java.io.*; 
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel; 
import java.util.*;
import java.lang.String;
import java.lang.Math;
import java.util.ArrayList;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


/// This application uses an HDFS inverted index to classify documents using kNN
/// Usage:
/// hadoop jar simir.jar kNN InvertedIndexFileName Train_List Test_List K
/// -- "InvertedIndexFileName" is the name (including path) of the HDFS inverted file
///     (Make sure that you have all the three files: 
///                 + InvertedIndexFileName.lex: lexicon
///                 + InvertedIndexFileName.pos: posting
///                 + InvertedIndexFileName.dlen: doc length
/// -- "Train_List" is the name (including path) of the file which contains training documnet IDs and their class tag
///                 It has the following format (each document at a separate line)
///                 Tag1 DocumentID1
///                 Tag2 DocumentID2
/// -- "Test_list" is the name (including path) of the file which contains testing documents
///                  It has the following format (each testing document at a separate line)
///        DocumentID1 term1 term2 ... termN
///        DocumentID2 termN+1 termN+2 ....
/// -- "K" is the value for the parameter k in kNN algorithm 
	

/// This is an auxiliary class for sorting documents based on their scores. 
class ValueComparator implements Comparator { 
 
    Map base; 
    public ValueComparator(Map base) { 
	this.base = base; 
    } 
    
    public int compare(Object a, Object b) { 
	
	if((Double)base.get(a) < (Double)base.get(b)) { 
	    return 1; 
	} else if((Double)base.get(a) == (Double)base.get(b)) { 
	    return 0; 
	} else { 
	    return -1; 
	} 
    } 
}

    

/// This is the main class for kNN.
public class kNN {

    static int TOTALCLASS = 20;
    /// This function returns the weight of a matched query term for a document
    /// rawTF: raw count of the matched query term in the document
    /// docFreq: document frequency of the matched term (i.e.., total number of documents in the collection
    ///                 that contain the term
    /// docCountTotal: total number of documents in the collection
    /// termCount: the total count of the term in the whole collection
    /// totalTermCount: the sum of the total count of *all* the terms in the collection
    /// docLength: length of the document (number of words)
    /// avgDocLength: average document length in the collection
    /// param: a retrieval parameter that can be set to any value through the third argument when executing "Retrieval" 
    static double weight(int rawTF, int docFreq, int docCountTotal, int termCount,int totalTermCount, int docLength, double avgDocLength, double param) {
	double idf = Math.log((1.0+docCountTotal)/(0.5+docFreq));
	return (rawTF*idf); 
	// this is the raw TF-IDF weighting, which ignored a lot of information 
	// passed to this weighitng function; you may explore different ways of 
	// exploiting all the information to see if you can improve retrieval accuracy.
	// BM25, Dirichlet prior, and Pivoted length normalization are obvious choices, 
	// but you are encouraged to think about other choices. 
    }
    
    /// This is the core function of the kNN algorithm
    /// sortedAcc: the ranked document list of the current test document, and the closet document ranks the highest
    /// trainTag: this hashmap stores the document IDs and their category tags from the training data
    /// numK: this is the k value for kNN
    static int categorization(TreeMap<String, Double> sortedAcc, HashMap<String, Integer> trainTag, int numK){
	/// initialize the vote counts for all the categories
	int [] counts = new int[TOTALCLASS];
	for(int j = 0; j < TOTALCLASS; j++){
	    counts[j] = 0;
	}

	/// Look up from the top ranked results until we find numK labeled documents
	int i=0;
	for (Map.Entry<String, Double> entry : sortedAcc.entrySet()) { 
	    String key = entry.getKey(); 

	    /// Look up the tag of the document
	    Integer currTag = trainTag.get(key);
	    if(currTag != null){
		//#########################################################//
		// add statements here so that after the loop, counts would 
		// have the votes from numK nearest neighbors for each category 
		// Hint: how to update "counts" and what other variables need to update? 
		// 
		//#########################################################//
		i=i+1;
		counts[currTag]=counts[currTag]+1;
	    }else{
		/// if one top ranked document does not have a label from the training data
		/// then just skip this document
		;
	    }
	    if (i >= numK) {
		break;
	    }
	}

	/// find the category with the largest count
	int maxClass = -1;
	int maxCount = 0;
	for(Integer j = 0; j < TOTALCLASS; j++){
	    if(counts[j] > maxCount){
		//#########################################################//
		// add statements here so that after the loop, maxClass would
		// have the tag of the category which has the largest number of votes
		// Hint: two variables need to update here? 
		// 
		//#########################################################//
		maxCount = counts[j];
		maxClass = j;
	    }
	}
	
	return maxClass;
	
    }


    public static void main (String [] args) throws IOException {
	

	/// This class defines the type Entry to pack all the information about a term stored in a lexicon entry. 
	class Entry {
	    public int df; // document frequency
	    public int count; // term count in the collection
	    public long pos; // start position of entries in the posting file
	    public int length; // span of postering entries 
	    Entry(int d, int c, long p, int l) {
		pos=p;
		length = l;
		df =d;
		count=c;
	    }
	}
	
	double retrievalModelParam = 0.5; // default retrieval parameter; this should be set to a meaningful value
	// for the retrieval model actually implemented. 

	// the following is standard HDFS setup 
	Configuration conf = new Configuration();
	FileSystem fs = FileSystem.get(conf);
	FSDataInputStream finlexicon=null, fintrain=null;
	FSDataInputStream  finposting=null, findoclen=null, finquery=null; 

	//Hash table for the lexicon:key is a term, value is an object of class Entry
	HashMap<String,Entry> lex= new HashMap<String,Entry>();

	// Hash table for the score accumulators: key is docID, value is score.
	HashMap<String,Double> acc = new HashMap<String,Double>();

	// Hash table for storing document length: key is docID, value is doc length
	HashMap<String,Integer> dlen = new HashMap<String,Integer>(); 

	// Hash table for storing the tags of training documents
	HashMap<String, Integer> trainTag = new HashMap<String, Integer>();

	Entry termEntry = null;
	byte [] buffer = null; 
	String docID =null;
	int termFreq; 
	StringTokenizer st=null;
	String term =null;
	int i; 
	double s; 

	int resultCount=1000; // this is the maximum number of results to return for each query
	int numK = 0; //this is the value of k for the kNN algorithm

	if (args.length>=5) {
	    retrievalModelParam = Double.parseDouble(args[4]); // parse the provided parameter value if available.
	}

	String t=null;
	BufferedReader reader = null;
	try { 
	    // open the three files for the index
	    finposting = fs.open(new Path(args[0] + ".pos" ));
	    finlexicon = fs.open(new Path(args[0] + ".lex"));
	    findoclen = fs.open(new Path(args[0] + ".dlen"));

	    // open the training tag file
	    fintrain = fs.open(new Path(args[1]));

	    // open the query file
	    finquery = fs.open(new Path(args[2])); 

	    // read the value of K
	    numK = Integer.parseInt(args[3]);
	} catch (IOException ioe) {
	    ioe.printStackTrace();
	    System.out.println("file operation error: " + "args[0]="+ args[0] + ";args[1]="+args[1]); 
            System.exit(1);
	}

	// load the lexicon 
	int totalTermCount=0;
	while (finlexicon.available()!=0) {
	    term = finlexicon.readUTF(); 
	    int docFreq = finlexicon.readInt();
	    int termCount =finlexicon.readInt();
	    long  startPos = finlexicon.readLong();
	    int postingSpan = finlexicon.readInt();
	    lex.put(term,new Entry(docFreq,termCount,startPos, postingSpan)); 
	    totalTermCount += termCount;
	}	    
	finlexicon.close();
	
	// load doc length
	double avgDocLen =0;
	int totalDocCount=0;
	reader = new BufferedReader(new InputStreamReader(findoclen));
	while ((t=reader.readLine()) != null) {
	    st = new StringTokenizer(t);	
	    term = st.nextToken();
	    int docLen = Integer.parseInt(st.nextToken().trim());
	    dlen.put(term,docLen);

	    // we'll use this opportunity to compute the average doc length and the total number of documents in the collection
	    // note that it's better to precompute these values in the indexing stage and store them in a file
	    avgDocLen += docLen;  
	    totalDocCount++;
	}
	avgDocLen /= totalDocCount; 
	findoclen.close(); 

	// load training tags
	reader = new BufferedReader(new InputStreamReader(fintrain));
	while ((t=reader.readLine()) != null) {
	    st = new StringTokenizer(t);
	    Integer currTag = Integer.parseInt(st.nextToken());
	    trainTag.put(st.nextToken(), currTag);
	}


	// process queries 
	reader = new BufferedReader(new InputStreamReader(finquery));
	while ((t=reader.readLine()) != null) {
	    // each line has precisely one query: queryID term1 term 2.... 

	    st = new StringTokenizer(t); // A StringTokenizer allows us to decompose a string into space-separated tokens
	    String qid = st.nextToken(); // the first token should be the query ID
	    System.err.println("Processing query:"+qid); 

	    acc.clear(); // clear the score accumulator to prepare for storing new scores for this query

	    int qlen=0; // counter for computing the query length

	    HashMap<String, Integer> qTermFreq = new HashMap<String, Integer>();

	    // trun the original query document into (term, freq) pairs
	    // this is to make the calculation faster
	    while (st.hasMoreTokens()) {
		term = st.nextToken();
		termEntry = lex.get(term);
		
		if (termEntry != null) {
		    qlen++;
		    Integer currFreq = qTermFreq.get(term);
		    if(currFreq == null){
			qTermFreq.put(term, 1);
		    }else{
			qTermFreq.put(term, currFreq+1);
		    }
		}
	    }

	    // iterate over all the terms in the query document
	    for(Map.Entry<String, Integer> entry : qTermFreq.entrySet()) {
		term = entry.getKey();
		termEntry = lex.get(term); // fetch the lexicon entry for this query term 

		if (termEntry != null) {
		    qlen++; 
		    int df = termEntry.df; 
		    // df tells us how many pairs (docID termCount) for this term we have in the posting file 

		    finposting.seek(termEntry.pos); // seek to the starting position of the posting entries for this term

		    for (i=1; i<=df; i++) { // read in the df pairs 
			docID = finposting.readUTF().trim(); // read in a document ID
			termFreq = finposting.readInt(); // read in the term Count 
			int doclen = dlen.get(docID).intValue(); // fetch the document length for this doc 
			double tmpWeight = weight(termFreq,df,totalDocCount,termEntry.count,totalTermCount,doclen,avgDocLen, retrievalModelParam);
			tmpWeight = tmpWeight * entry.getValue();
			// compute the weight of this matched term

			Double s1 = acc.get(docID); // get the current score for this docID in the accumulator if any
			if (s1 != null) { 
			    // this means that the docID already has an entry in the accumulator, i.e., the docID already matched a previous query term
			    acc.put(docID, s1.doubleValue() + tmpWeight);
			    
			} else {
			    // otherwise, we need to add a score accumulator for this docID and set the score appropriately.
			    acc.put(docID, tmpWeight);
			}
		    }
		    
		} else{
		    System.err.println("Skipping query term:"+term+ "(not in the collection)");
		}
	    }
	    

	    // At this point, we have iterated over all the query terms and updated the score accumulators appropriately
	    // so the score accumulators should have a sum of weights for all the matched query terms. 
	    // In some retrieval models, we may need to adjust this sum in some way, we can do it here

	    // adjustment of scores for each document if necessary
	    for (Map.Entry<String, Double> entry : acc.entrySet()) { 
		// iterate over all accumulators and use "acc.put()" to update the score
		// for example, the following statement would add |Q| log (mu/(mu+|D|)), which is needed for Dirichlet prior 
		/* 
		acc.put(entry.getKey(), entry.getValue().doubleValue()+
		qlen*Math.log(retrievalModelParam/(retrievalModelParam+dlen.get(docID).intValue()))); */
	    }

	    // now we've finished scoring, and we'll sort the scores and output the top N results 
	    // to the standard output stream (System.out)
	    ValueComparator bvc =  new ValueComparator(acc); 
	    TreeMap<String,Double> sortedAcc = new TreeMap<String,Double>(bvc); 
 	    sortedAcc.putAll(acc); 

	    // call the core function of kNN algorithm
	    int resTag = categorization(sortedAcc, trainTag, numK);

	    // print out the classification results
	    System.out.println(resTag + " " + qid);

	}//end of a single query
    }
}


