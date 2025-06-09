import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Serializable;
import scala.Tuple2;

import java.util.*;
import java.util.concurrent.Semaphore;
import java.util.stream.Collectors;

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
        HashMap<Long, Long> streamItemsFrequency = new HashMap<>();
        Sketch streamCountSketch = new Sketch(D, W);

        // initializing the hash functions
        HashFunction[] hashFunctions = methodsHW3.generateHashFunctions(D, W);
        HashFunction[] randomBias = methodsHW3.generateHashFunctions(D, 2);

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
                            // true frequencies
                            Map<Long, Long> batchItemsFrequencies =  methodsHW3.countDistinctItemsFreq(batch);

                            // Count-min Sketch
                            //

                            // Count Sketch
                            Sketch batchCountSketch = methodsHW3.ComputeCountSketch(batch, D, W, hashFunctions, randomBias);

                            // exporting both Count-min Sketch and Count Sketch
                            if(!streamCountSketch.mergeWith(batchCountSketch)) System.err.println("Merge failed");

                            // exporting the true frequencies of each distinct item
                            for (Map.Entry<Long, Long> pair : batchItemsFrequencies.entrySet()) {
                                if (!streamItemsFrequency.containsKey(pair.getKey())) streamItemsFrequency.put(pair.getKey(), pair.getValue());
                                else if(!Objects.equals(streamItemsFrequency.get(pair.getKey()), pair.getValue())) streamItemsFrequency.replace(pair.getKey(), streamItemsFrequency.get(pair.getKey()) + pair.getValue());
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
        Map<Long, Long> streamItemsFrequenciesCS = methodsHW3.getItemFrequencies(streamItemsFrequency, streamCountSketch, hashFunctions, randomBias);
        double avgFreqErrorsCS = methodsHW3.calcAvgErrorKhittersCS(streamItemsFrequency, streamItemsFrequenciesCS, K);

        System.out.println("Number of processed items = " + streamLength[0]);
        System.out.println("Number of distinct items = " + streamItemsFrequency.size());
        System.out.println("Number of Top-K heavy hitters = " + streamItemsFrequency.size());
        long max = 0L;
        ArrayList<Long> distinctKeys = new ArrayList<>(streamItemsFrequency.keySet());
        Collections.sort(distinctKeys, Collections.reverseOrder());
        System.out.println("Largest item = " + distinctKeys.get(0));
    }
}

class HashFunction implements Serializable {
    int a;
    int b;
    int p;
    int C;

    HashFunction(int codomainSize){
        a = 0;
        b = 1;
        p = 8121;
        C = codomainSize;
    }

    public void setRandomCoeffs(){
        Random randomGenerator = new Random();
        do{ a = randomGenerator.nextInt(p); } while (a == 0);
        b = randomGenerator.nextInt(p);
    }

    public int computeHashValue(Long x){
        return ((int) ((a*x+b) % p) % C);
    }
}

class Sketch implements Serializable {
    int rowsSketch;
    int colsSketch;
    long[][] table;

    Sketch(int rowsSketch, int colsSketch){
        this.rowsSketch = rowsSketch;
        this.colsSketch = colsSketch;
        table = new long[rowsSketch][colsSketch];
        for(int i = 0; i < rowsSketch; i++) for(int j = 0; j < colsSketch; j++) setCounter(i, j, 0);
    }

    public long getCounter(int row, int col){
        return table[row][col];
    }

    public void setCounter(int row, int col, long value){
        table[row][col] = value;
    }

    public boolean mergeWith(Sketch sketch){
        if(sketch.getRowsSketch() != getRowsSketch() || sketch.getColsSketch() != getColsSketch()) return false;
        for(int j = 0; j < sketch.getRowsSketch(); j++){
            for(int c = 0; c < sketch.getColsSketch(); c++){
                table[j][c] += sketch.getCounter(j, c);
            }
        }
        return true;
    }

    static public Sketch merge(Sketch a, Sketch b){
        Sketch destSketch = new Sketch(a.rowsSketch, a.colsSketch);
        for(int i = 0; i < destSketch.getRowsSketch(); i++) for(int j = 0; j < destSketch.getColsSketch(); j++) destSketch.setCounter(i, j, 0);
        for(int j = 0; j < b.getRowsSketch(); j++){
            for(int c = 0; c < b.getColsSketch(); c++){
                destSketch.setCounter(j, c, a.getCounter(j, c) + b.getCounter(j, c));
            }
        }
        return destSketch;
    }

    public int getRowsSketch(){
        return rowsSketch;
    }

    public int getColsSketch(){
        return colsSketch;
    }
}

class methodsHW3{
    static public Map<Long, Long> getDistinctItems(JavaRDD<String> batch){
        return batch
                .mapToPair(s -> new Tuple2<>(Long.parseLong(s), 1L)) // associate each item with count 1
                .reduceByKey((i1, i2) -> 1L) // removes duplicates of the same key
                .collectAsMap();
    }

    static public Map<Long, Long> countDistinctItemsFreq(JavaRDD<String> batch){
        // Extract the distinct items and counts from the batch
        Map<Long, Long> batchItems = batch
                .mapToPair(s -> new Tuple2<>(Long.parseLong(s), 1L)) // associate each item with count 1
                .reduceByKey(Long::sum) // removes duplicates of the same key by summing up their counts
                .collectAsMap(); // return distinct items: item-count
        return batchItems;
    }

    static public Sketch ComputeCountSketch(JavaRDD<String> batch, int rowsSketch, int colsSketch, HashFunction[] hashFunctions, HashFunction[] randomBias){
        /* PREVIOUS VERSION: BATCH AS COLLECTION
        CountSketch batchSketch = new CountSketch(rowsSketch, colsSketch);
        List<Long> items = batch.map(Long::parseLong).collect();
        for(Long item : items){
            for (int j = 0; j < rowsSketch; j++) {
                int hj = hashFunctions[j].computeHashValue(item);
                int randomChange = randomBias[j].computeHashValue(item);
                if (randomChange == 0) randomChange = -1;
                sketch.setCounter(j, hj, (sketch.getCounter(j, hj) + randomChange));
            }
        }*/

        // rdd VERSION: BATCH AS PARTITIONS of RDD'S (way more space-efficient)
        Sketch batchSketch = batch.mapPartitions(partition -> {
            Sketch partitionSketch = new Sketch(rowsSketch, colsSketch);

            partition.forEachRemaining(batchItem ->{
                Long item = Long.parseLong(batchItem);
                for (int j = 0; j < rowsSketch; j++) {
                    int hj = hashFunctions[j].computeHashValue(item);
                    int randomChange = randomBias[j].computeHashValue(item);
                    if (randomChange == 0) randomChange = -1;
                    partitionSketch.setCounter(j, hj, (partitionSketch.getCounter(j, hj) + randomChange));
                }
            });
            return Collections.singleton(partitionSketch).iterator();
        }).reduce(Sketch::merge);

        return batchSketch;
    }

    static public Map<Long, Long> getItemFrequencies(Map<Long, Long> distinctItems, Sketch sketch, HashFunction[] hashFunctions, HashFunction[] randomBias){
        HashMap<Long, Long> itemsMedianFreq = new HashMap<>();
        for(Long key : distinctItems.keySet()){
            long[] itemFreqs = new long[sketch.getRowsSketch()];
            for(int j = 0; j < sketch.getRowsSketch(); j++){
                int hj = hashFunctions[j].computeHashValue(key);
                int randomChange = randomBias[j].computeHashValue(key);
                if(randomChange == 0) randomChange = -1;
                itemFreqs[j] = randomChange * sketch.getCounter(j, hj);
            }
            itemsMedianFreq.put(key, getMedianFrequency(itemFreqs));
        }
        return itemsMedianFreq;
    }

    static public HashFunction[] generateHashFunctions(int rowsSketch, int colsSketch){
        HashFunction[] hashFunctions = new HashFunction[rowsSketch];
        for(int i = 0; i < rowsSketch; i++){
            hashFunctions[i] = new HashFunction(colsSketch);
            hashFunctions[i].setRandomCoeffs();
        }
        return hashFunctions;
    }

    static private long getMedianFrequency(long[] array){
        long[] arr = Arrays.copyOf(array, array.length);
        Arrays.sort(arr);
        int mid = arr.length/2;
        if(arr.length % 2 == 0) return Math.round((double) (arr[(mid) - 1] + arr[mid])/2.0);
        else return arr[mid];
    }

    static public double calcAvgErrorKhittersCS(Map<Long, Long> itemFreqs, Map<Long, Long> freqsItemCS, int K){
        // getting the K-Hitters for the true frequencies
        List<Tuple2<Long, Long>> trueKhitters = getTrueKhitters(itemFreqs, K);

        // getting the K-Hitters for the estimates
        List<Tuple2<Long, Long>> kHittersCS = new ArrayList<>();
        for(Tuple2<Long, Long> item : trueKhitters) kHittersCS.add(new Tuple2<>(item._1(), freqsItemCS.get(item._1())) );

        // computing the average relative error between estimate and true freq
        // because of how we built the second array, we expect the items to have the same index in both of them
        double cumulativeError = 0;
        for(int i = 0; i < trueKhitters.size(); i++){
            long trueFreq = trueKhitters.get(i)._2();
            long estimatedFreq = kHittersCS.get(i)._2();
            cumulativeError += (double) Math.abs(trueFreq - estimatedFreq) / trueFreq;
        }
        double avgErrorCS = cumulativeError / trueKhitters.size();
        return avgErrorCS;
    }

    static public List<Tuple2<Long, Long>> getTrueKhitters(Map<Long, Long> itemFreqs, int K){
        ArrayList<Tuple2<Long, Long>> trueFreqs = new ArrayList<>();
        for (Map.Entry<Long, Long> e : itemFreqs.entrySet()) {
            trueFreqs.add(new Tuple2<>(e.getKey(), e.getValue()));
        }
        Comparator<Tuple2<Long, Long>> descendingLongComparator = Comparator.comparingLong(Tuple2::_2);
        trueFreqs.sort(descendingLongComparator.reversed());
        final long freqThreshold = trueFreqs.get(K-1)._2();
        List<Tuple2<Long, Long>> trueKhitters = trueFreqs.stream().filter(t -> t._2() >= freqThreshold).collect(Collectors.toList());
        return trueKhitters;
    }
}