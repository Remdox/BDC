import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

import java.util.*;
import java.util.concurrent.Semaphore;

public class G13HW3 {
    // public static final int THRESHOLD = 1000000;

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            throw new IllegalArgumentException("USAGE: port, threshold");
        }
        // IMPORTANT: the master must be set to "local[*]" or "local[n]" with n > 1, otherwise
        // there will be no processor running the streaming computation and your
        // code will crash with an out of memory (because the input keeps accumulating).
        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]") // remove this line if running on the cluster
                .setAppName("DistinctExample");

        // The definition of the streaming spark context  below, specifies the amount of
        // time used for collecting a batch, hence giving some control on the batch size.
        // Beware that the data generator we are using is very fast, so the suggestion is to
        // use batches of less than a second, otherwise you might exhaust the JVM memory.
        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        Semaphore stoppingSemaphore = new Semaphore(1);
        stoppingSemaphore.acquire();

        int portExp = Integer.parseInt(args[0]);
        System.out.println("Receiving data from port = " + portExp);
        int T = Integer.parseInt(args[1]); // threshold
        System.out.println("Threshold = " + T);
        int D = Integer.parseInt(args[2]); // number of rows of each sketch
        int W = Integer.parseInt(args[3]); // number of cols of each sketch
        int K = Integer.parseInt(args[4]); // number of top frequent items of interest

        // Variable streamLength below is used to maintain the number of processed stream items.
        // It must be defined as a 1-element array so that the value stored into the array can be
        // changed within the lambda used in foreachRDD. Using a simple external counter streamLength of type
        // long would not work since the lambda would not be allowed to update it.
        long[] streamLength = new long[1]; // Stream length (an array to be passed by reference)
        streamLength[0] = 0L;
        HashMap<Long, Long> histogram = new HashMap<>();

        // CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
                // BEWARE: the `foreachRDD` method has "at least once semantics", meaning
                // that the same data might be processed multiple times in case of failure.
                .foreachRDD((batch, time) -> {
                    // this is working on the batch at time `time`.
                    if (streamLength[0] < T) {
                        long batchSize = batch.count();
                        streamLength[0] += batchSize;
                        if (batchSize > 0) {
                            System.out.println("Batch size at time [" + time + "] is: " + batchSize);
                            Map<Long, Long> batchItems =  methodsHW3.countDistinctItems(batch);
                            for (Map.Entry<Long, Long> pair : batchItems.entrySet()) {
                                if (!histogram.containsKey(pair.getKey())) {
                                    histogram.put(pair.getKey(), pair.getValue());
                                }
                            }
                            // If we wanted, here we could run some additional code on the global histogram

                            if (streamLength[0] >= T) {
                                // Stop receiving and processing further batches
                                stoppingSemaphore.release();
                            }

                        }
                    }
                });

        // MANAGING STREAMING SPARK CONTEXT
        System.out.println("Starting streaming engine");
        sc.start();
        System.out.println("Waiting for shutdown condition");
        stoppingSemaphore.acquire();
        System.out.println("Stopping the streaming engine");
        sc.stop(false, false);
        System.out.println("Streaming engine stopped");

        // COMPUTE AND PRINT FINAL STATISTICS
        System.out.println("Number of items processed = " + streamLength[0]);
        System.out.println("Number of distinct items = " + histogram.size());
        long max = 0L;
        ArrayList<Long> distinctKeys = new ArrayList<>(histogram.keySet());
        Collections.sort(distinctKeys, Collections.reverseOrder());
        System.out.println("Largest item = " + distinctKeys.get(0));
    }
}

class methodsHW3{
    static public Map<Long, Long> countDistinctItems(JavaRDD<String> batch){
        // Extract the distinct items and counts from the batch
        Map<Long, Long> batchItems = batch
                .mapToPair(s -> new Tuple2<>(Long.parseLong(s), 1L)) // associate each item with count 1
                .reduceByKey(Long::sum) // removes duplicates of the same key
                .collectAsMap(); // return distinct items: item-1
        return batchItems;
    }
}