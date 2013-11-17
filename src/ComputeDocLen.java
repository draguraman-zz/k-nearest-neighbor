
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import java.util.HashMap;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * This is an Hadoop Map/Reduce application that computes document lengths based on 
 * the "raw inverted index" (i.e., output generated by "InvertedIndex").  
 *
 * To run: hadoop jar simir.jar ComputeDocLen 
 *            [-m <i>maps</i>] [-r <i>reduces</i>] <i>in-dir</i> <i>out-dir</i> 
 *   "in-dir" has all the raw inverted index files generated by "InvertedIndex"
 *   "out-dir" is the directory to put the document length table. 
 */
public class ComputeDocLen extends Configured implements Tool {
  
  /**
   *
   * For each line of input, skip the first string (the term), then read each pair (docID, termCount),
   * and emit (<b>docID</b>, <b>termCount</b>).
   */
  public static class MapClass extends MapReduceBase
    implements Mapper<LongWritable, Text, Text, Text> {
    
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
      private Text did = new Text(); 
    
    public void map(LongWritable key, Text value, 
                    OutputCollector<Text, Text> output, 
                    Reporter reporter) throws IOException {
      String line = value.toString();
      StringTokenizer itr = new StringTokenizer(line);
      String docID ="";
      String term ="";
     
      if (itr.hasMoreTokens()) {
	  term=itr.nextToken();
	  while (itr.hasMoreTokens()) {
	      docID = itr.nextToken(); 
	      did.set(docID);
	      word.set(itr.nextToken());
	      output.collect(did,word);
	  }
      }
    }
  }
  
  /**
   * A reducer class that just emits the sum of the input values.
   */
  public static class Reduce extends MapReduceBase
    implements Reducer<Text, Text, Text, Text> {
    
      Text s= new Text(); 
    public void reduce(Text key, Iterator<Text> values,
                       OutputCollector<Text, Text> output, 
                       Reporter reporter) throws IOException {
	String sum = "";
	int count=0;
	while (values.hasNext()) {
	    count = count + Integer.parseInt(values.next().toString().trim()) ;
	}
	Text t = new Text(); 
	t.set(Integer.toString(count));
	output.collect(key,t);
    }
  }
  
  static int printUsage() {
    System.out.println("ComputeDocLen [-m <maps>] [-r <reduces>] <input> <output>");
    ToolRunner.printGenericCommandUsage(System.out);
    return -1;
  }
  
  /**
   * The main driver for ComputeDocLen map/reduce program.
   * Invoke this method to submit the map/reduce job.
   * @throws IOException When there is communication problems with the 
   *                     job tracker.
   */
  public int run(String[] args) throws Exception {
    JobConf conf = new JobConf(getConf(), ComputeDocLen.class);
    conf.setJobName("computerdoclength");
 
    // the keys are words (strings)
    conf.setOutputKeyClass(Text.class);
    // the values are strings too
    conf.setOutputValueClass(Text.class);
    
    conf.setMapperClass(MapClass.class);        
    conf.setCombinerClass(Reduce.class);
    conf.setReducerClass(Reduce.class);
    
    List<String> other_args = new ArrayList<String>();
    for(int i=0; i < args.length; ++i) {
      try {
        if ("-m".equals(args[i])) {
          conf.setNumMapTasks(Integer.parseInt(args[++i]));
        } else if ("-r".equals(args[i])) {
          conf.setNumReduceTasks(Integer.parseInt(args[++i]));
        } else {
          other_args.add(args[i]);
        }
      } catch (NumberFormatException except) {
        System.out.println("ERROR: Integer expected instead of " + args[i]);
        return printUsage();
      } catch (ArrayIndexOutOfBoundsException except) {
        System.out.println("ERROR: Required parameter missing from " +
                           args[i-1]);
        return printUsage();
      }
    }
    // Make sure there are exactly 2 parameters left.
    if (other_args.size() != 2) {
      System.out.println("ERROR: Wrong number of parameters: " +
                         other_args.size() + " instead of 2.");
      return printUsage();
    }
    FileInputFormat.setInputPaths(conf, other_args.get(0));
    FileOutputFormat.setOutputPath(conf, new Path(other_args.get(1)));
        
    JobClient.runJob(conf);
    return 0;
  }
  
  
  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new ComputeDocLen(), args);
    System.exit(res);
  }

}