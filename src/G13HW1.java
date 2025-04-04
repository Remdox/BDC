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
    public static void MRComputeStandardObjective(JavaRDD<Vector> parsedInputPoints, Vector[] C){
        double distance = 0.0;
        long count = parsedInputPoints.count();
        for(int i=0; i<count; i++){
            Vector point = parsedInputPoints.take(i+1).get(i); // TODO da cambiare
            double minDistance = 0;
            Vector minC = Vectors.zeros(2); // TODO da cambiare
            for(int j=0; j<C.length; j++){
                double distance1 = Vectors.sqdist(point, C[j]);
                if(distance1 < minDistance){
                    minDistance = distance1;
                    minC = C[j];
                }
            }
            distance += Vectors.sqdist(point, minC);
        }
        System.out.println(distance);
        double Delta = (1.0 / count) * distance;
        System.out.println("Delta: " + Delta);
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

        System.out.println(args[0] + " " + args[1] + " " + args[2] + " " + args[3]);
        int L = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int M = Integer.parseInt(args[3]);
        JavaRDD<String> inputPoints = sc.textFile(args[0]).repartition(L).cache();

        long N = inputPoints.count();
        System.out.println("Number of points: " + N);
        long NA = inputPoints.filter(row -> row.endsWith("A")).count();
        long NB = inputPoints.filter(row -> row.endsWith("B")).count();
        System.out.println("Number of points in A: " + NA);
        System.out.println("Number of points in B: " + NB);

        JavaRDD<Vector> parsedInputPoints = inputPoints.map(row -> {
            String[] rowArray = row.split(",");
            double[] values = new double[rowArray.length - 1];
            for (int i = 0; i < rowArray.length - 1; i++) {
                values[i] = Double.parseDouble(rowArray[i]);
            }
            return Vectors.dense(values);
        });
        parsedInputPoints.cache();

        KMeansModel clusters = KMeans.train(parsedInputPoints.rdd(), K, M);

        System.out.println("Cluster centers: ");
        Vector[] C = clusters.clusterCenters();
        for (Vector center: C) {
            System.out.println(center);
        }
        myMethod.MRComputeStandardObjective(parsedInputPoints, C);

    }
}
