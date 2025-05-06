import org.apache.datasketches.memory.Memory;
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
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import java.util.*;
import java.lang.*;

class methodsHW2{
    public static void MRFairLloyd(JavaRDD<String> inputPoints, Integer K, Integer M){
        JavaRDD<Tuple2<Vector, String>> parsedInputPoints = conversionWithGroups(inputPoints).persist(StorageLevel.MEMORY_ONLY_SER()); // TODO: fare una prova sul tempo che ci mette con caching serializzato
        KMeansModel clusters = KMeans.train(parsedInputPoints.map(Tuple2::_1).rdd(), K, 0);
        Vector[] C = clusters.clusterCenters();

        for(int i = 0; i < M; i++){ // M iterations of Fair K-Means Clustering
            JavaPairRDD<Vector, Iterable<Tuple2<Vector, String>>> pairsPointsToCluster = parsedInputPoints.mapToPair(point -> {
                int clusterIndex = clusters.predict(point._1()); // returns index of the corresponding cluster that includes the point
                Vector centroid = C[clusterIndex];
                return new Tuple2<>(centroid, point); // returns a pair point-center of the corresponding cluster

            }).groupByKey(); // groups points of the same cluster
            JavaPairRDD<Vector, Iterable<Tuple2<Vector, String>>> clustersGroupA = getClustersOfGroup(pairsPointsToCluster, "A");
            JavaPairRDD<Vector, Iterable<Tuple2<Vector, String>>> clustersGroupB = getClustersOfGroup(pairsPointsToCluster, "B");


            Vector[] centersA = {};
            Vector[] centersB = {};

        }

    }

    public static JavaPairRDD<Vector, Iterable<Tuple2<Vector, String>>> getClustersOfGroup(JavaPairRDD<Vector, Iterable<Tuple2<Vector, String>>> pairsPointsToCluster, String group) {
        return pairsPointsToCluster.mapValues(cluster -> {
            ArrayList<Tuple2<Vector, String>> groupPointsInCluster = new ArrayList<>();
            for(Tuple2<Vector, String> tuple : cluster){
                if(tuple._2().endsWith(group)) groupPointsInCluster.add(tuple);
            }
            return groupPointsInCluster;
        });
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

    public static JavaRDD<Tuple2<Vector, String>>  conversionWithGroups(JavaRDD<String> inputPoints){
        JavaRDD<Tuple2<Vector, String>> parsedInputPoints = inputPoints.map(row -> {
            String[] rowArray = row.split(",");
            double[] values = new double[rowArray.length - 1];
            for (int i = 0; i < rowArray.length - 1; i++) {
                values[i] = Double.parseDouble(rowArray[i]);
            }
            return new Tuple2<>(Vectors.dense(values), rowArray[rowArray.length - 1]);
        });
        parsedInputPoints.cache();

        return parsedInputPoints;
    }

    public static double[] computeVectorX(double fixedA, double fixedB, double[] alpha, double[] beta, double[] ell, int K) {
        double gamma = 0.5;
        double[] xDist = new double[K];
        double fA, fB;
        double power = 0.5;
        int T = 10;
        for (int t=1; t<=T; t++){
            fA = fixedA;
            fB = fixedB;
            power = power/2;
            for (int i=0; i<K; i++) {
                double temp = (1-gamma)*beta[i]*ell[i]/(gamma*alpha[i]+(1-gamma)*beta[i]);
                xDist[i]=temp;
                fA += alpha[i]*temp*temp;
                temp=(ell[i]-temp);
                fB += beta[i]*temp*temp;
            }
            if (fA == fB) {break;}
            gamma = (fA > fB) ? gamma+power : gamma-power;
        }
        return xDist;
    }

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
}

public class G13HW2 {
    static void main(String[] args) {
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
        JavaRDD<Vector> parsedInputPoints = methodsHW2.conversion(inputPoints);

        KMeansModel clusters = KMeans.train(parsedInputPoints.rdd(), K, M);
        Vector[] C_stand = clusters.clusterCenters();
    }
}
