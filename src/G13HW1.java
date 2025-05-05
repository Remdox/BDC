import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.*;
import scala.Tuple2;
import java.util.*;
import java.lang.*;

class methodsHW1{
    public static double MRComputeStandardObjective(JavaRDD<Vector> parsedInputPoints, Vector[] C){
        double totalDistance = parsedInputPoints.map( point -> {
            double minDistance = Double.MAX_VALUE;
            for (Vector c : C) {
                double distance = Vectors.sqdist(point, c);
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            return minDistance;
        }).reduce(Double::sum);
        return (1.0 / parsedInputPoints.count()) * totalDistance;
    }

    public static double MRComputeFairObjective(JavaRDD<String> inputPoints, Vector[] C){
        JavaRDD<String> inputPointsA = inputPoints.filter(row -> row.endsWith("A"));
        JavaRDD<Vector> parsedInputPointsA = conversion(inputPointsA);
        double deltaA = MRComputeStandardObjective(parsedInputPointsA, C);

        JavaRDD<String> inputPointsB = inputPoints.filter(row -> row.endsWith("B"));
        JavaRDD<Vector> parsedInputPointsB = conversion(inputPointsB);
        double deltaB = MRComputeStandardObjective(parsedInputPointsB, C);

        return Double.max(deltaA, deltaB);
    }

    public static Map<Vector, Long> mapClusterSizesToCenter(JavaRDD<Vector> parsedInputPoints, Vector[] C, Integer L){
        Random randomGenerator = new Random();
        JavaPairRDD<Vector, Long> N = parsedInputPoints.flatMapToPair(point ->{
            HashMap<Vector, Long> counts = new HashMap<>();
            ArrayList<Tuple2<Vector, Long>> pairs = new ArrayList<>();
            double minDistance = Double.MAX_VALUE;
            Vector minC = Vectors.zeros(C.length);
            for (Vector c : C) {
                double distance = Vectors.sqdist(point, c);
                if (distance < minDistance) {
                    minDistance = distance;
                    minC = c;
                }
            }
            counts.put(minC, 1L + counts.getOrDefault(minC, 0L));
            for (Map.Entry<Vector, Long> e : counts.entrySet()) {
                pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
            }
            return pairs.iterator();
        }).groupBy((wordCountPair) -> randomGenerator.nextInt(L))
        .flatMapToPair(element ->{
            HashMap<Vector, Long> counts = new HashMap<>();
            for (Tuple2<Vector, Long> c : element._2()) {
                counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L));
            }
            ArrayList<Tuple2<Vector, Long>> pairs = new ArrayList<>();
            for (Map.Entry<Vector, Long> e : counts.entrySet()) {
                pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
            }
            return pairs.iterator();
        })
        .reduceByKey(Long::sum);
        return N.collectAsMap();
    }

    public static void MRPrintStatistics(JavaRDD<String> inputPoints, Vector[] C, Integer L){
        JavaRDD<String> inputPointsA = inputPoints.filter(row -> row.endsWith("A"));
        JavaRDD<Vector> parsedInputPointsA = conversion(inputPointsA);

        JavaRDD<String> inputPointsB = inputPoints.filter(row -> row.endsWith("B"));
        JavaRDD<Vector> parsedInputPointsB = conversion(inputPointsB);

        Map<Vector, Long> NA2 = mapClusterSizesToCenter(parsedInputPointsA, C, L);
        Map<Vector, Long> NB2 = mapClusterSizesToCenter(parsedInputPointsB, C, L);

        double[] centerCoordinates;
        for(int i=0; i<C.length; i++){
            centerCoordinates = C[i].toArray();
            System.out.println("i = " + i + ", center = (" +  String.format("%.6f", centerCoordinates[0]) + ", " + String.format("%.6f", centerCoordinates[1]) + "), NA" + i + "= " + NA2.getOrDefault(C[i], 0L) + ", NB" + i + "= " + NB2.getOrDefault(C[i], 0L));
        }
    }

    public static JavaRDD<Vector> conversion(JavaRDD<String> inputPoints){
        JavaRDD<Vector> parsedInputPoints = inputPoints.map(row -> {
            String[] rowArray = row.split(",");
            double[] values = new double[rowArray.length - 1];
            for (int i = 0; i < rowArray.length - 1; i++) {
                values[i] = Double.parseDouble(rowArray[i]);
            }
            return Vectors.dense(values);
        });
        parsedInputPoints.cache();

        return parsedInputPoints;
    }
}

public class G13HW1 {
    public static void main(String[] args) {
        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: file_path L K M");
        }

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("G13HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        System.out.println("Input file = " + args[0] + ", L = " + args[1] + ", K = " + args[2] + ", M = " + args[3]);
        int L = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int M = Integer.parseInt(args[3]);

        // partition the points into L random partitions for the MapReduce algorithm used later in MRPrintStatistics
        JavaRDD<String> inputPoints = sc.textFile(args[0]).repartition(L).cache();

        // as requested, we count the sizes of the two groups
        long N = inputPoints.count();
        long NA = inputPoints.filter(row -> row.endsWith("A")).count();
        long NB = inputPoints.filter(row -> row.endsWith("B")).count();
        System.out.println("N = " + N + ", NA= " + NA + ", NB = " + NB);

        // as requested, we first convert the strings given as input into Vector points
        JavaRDD<Vector> parsedInputPoints = methodsHW1.conversion(inputPoints);

        KMeansModel clusters = KMeans.train(parsedInputPoints.rdd(), K, M);
        Vector[] C = clusters.clusterCenters();

        double delta = methodsHW1.MRComputeStandardObjective(parsedInputPoints, C);
        System.out.println("Delta(U, C) = " + String.format("%.6f", delta));

        double phi = methodsHW1.MRComputeFairObjective(inputPoints, C);
        System.out.println("Phi(A, B, C) = " + String.format("%.6f", phi));

        methodsHW1.MRPrintStatistics(inputPoints, C, L);
    }
}
