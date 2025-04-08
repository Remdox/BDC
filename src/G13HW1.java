import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.*;
import scala.Tuple2;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.lang.*;

class myMethod{
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

    public static Map<Vector, Integer> mapClusterSizesToCenter(JavaRDD<Vector> parsedInputPoints, Vector[] C){
        JavaPairRDD<Vector, Integer> N = parsedInputPoints.flatMapToPair(point ->{
            HashMap<Vector, Integer> counts = new HashMap<>();
            ArrayList<Tuple2<Vector, Integer>> pairs = new ArrayList<>();
            double minDistance = Double.MAX_VALUE;
            Vector minC = Vectors.zeros(C.length);
            for (Vector c : C) {
                double distance = Vectors.sqdist(point, c);
                if (distance < minDistance) {
                    minDistance = distance;
                    minC = c;
                }
            }
            counts.put(minC, 1 + counts.getOrDefault(minC, 0));
            for (Map.Entry<Vector, Integer> e : counts.entrySet()) {
                pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
            }
            return pairs.iterator();
        }).reduceByKey((x, y) -> x+y);
        return N.collectAsMap();
    }

    public static void MRPrintStatistics(JavaRDD<String> inputPoints, Vector[] C){
        JavaRDD<String> inputPointsA = inputPoints.filter(row -> row.endsWith("A"));
        JavaRDD<Vector> parsedInputPointsA = conversion(inputPointsA);

        JavaRDD<String> inputPointsB = inputPoints.filter(row -> row.endsWith("B"));
        JavaRDD<Vector> parsedInputPointsB = conversion(inputPointsB);

        Map<Vector, Integer> NA2 = mapClusterSizesToCenter(parsedInputPointsA, C);
        Map<Vector, Integer> NB2 = mapClusterSizesToCenter(parsedInputPointsB, C);

        double[] centerCoordinates;
        for(int i=0; i<C.length; i++){
            centerCoordinates = C[i].toArray();
            System.out.println("i = " + i + ", center = (" +  String.format("%.6f", centerCoordinates[0]) + " " + String.format("%.6f", centerCoordinates[1]) + "), NA" + i + "= " + NA2.getOrDefault(C[i], 0) + ", NB" + i + "= " + NB2.getOrDefault(C[i], 0));
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
    public static void main(String[] args) throws IOException {
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
        JavaRDD<String> inputPoints = sc.textFile(args[0]).repartition(L).cache();

        long N = inputPoints.count();
        long NA = inputPoints.filter(row -> row.endsWith("A")).count();
        long NB = inputPoints.filter(row -> row.endsWith("B")).count();
        System.out.println("N = " + N + ", NA= " + NA + ", NB = " + NB);

        JavaRDD<Vector> parsedInputPoints = myMethod.conversion(inputPoints);

        KMeansModel clusters = KMeans.train(parsedInputPoints.rdd(), K, M);
        Vector[] C = clusters.clusterCenters();
        /*Vector[] C = new Vector[4];
        C[0] = Vectors.dense(40.749035, -73.984431);
        C[1] = Vectors.dense(40.873440,-74.192170);
        C[2] = Vectors.dense(40.693363,-74.178147);
        C[3] = Vectors.dense(40.746095,-73.830627);*/

        double delta = myMethod.MRComputeStandardObjective(parsedInputPoints, C);
        System.out.println("Delta(U, C) = " + String.format("%.6f", delta));

        double phi = myMethod.MRComputeFairObjective(inputPoints, C);
        System.out.println("Phi(A, B, C) = " + String.format("%.6f", phi));

        myMethod.MRPrintStatistics(inputPoints, C);
    }
}
