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
                    for(int i=0; i<C.length; i++){
                        double distance = Vectors.sqdist(point, C[i]);
                        if(distance < minDistance){
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

    /*public static void MRPrintStatistics(JavaRDD<String> inputPoints, Vector[] C){
        JavaRDD<String> inputPointsA = inputPoints.filter(row -> row.endsWith("A"));
        JavaRDD<Vector> parsedInputPointsA = conversion(inputPointsA);

        JavaRDD<String> inputPointsB = inputPoints.filter(row -> row.endsWith("B"));
        JavaRDD<Vector> parsedInputPointsB = conversion(inputPointsB);

        JavaPairRDD<Vector, Vector> ciao = parsedInputPointsA.map(point ->{
            double minDistance = Double.MAX_VALUE;
            Vector minC = Vectors.zeros(2);
            for(int i=0; i<C.length; i++){
                double distance = Vectors.sqdist(point, C[i]);
                if(distance < minDistance){
                    minDistance = distance;
                    minC = C[i];
                }
            }
            return point, minC;
        });

        for(int i=0; i<C.length; i++){
            System.out.println("i = " + i + ", center = " + C[i]);
        }
    }*/

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

        //double cost = (1.0 / parsedInputPoints.count()) * clusters.computeCost(parsedInputPoints.rdd());
        //System.out.println("Cost = " + cost); // Per controllare Delta
        double Delta = myMethod.MRComputeStandardObjective(parsedInputPoints, C);
        System.out.println("Delta(U, C) = " + Delta);

        double Phi = myMethod.MRComputeFairObjective(inputPoints, C);
        System.out.println("Phi(A, B, C) = " + Phi);

        //myMethod.MRPrintStatistics(inputPoints, C);
    }
}
