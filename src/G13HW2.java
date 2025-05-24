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

import static java.lang.Math.sqrt;

public class G13HW2 {
    public static void main(String[] args) {
        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: file_path L K M");
        } // unfairDataset.csv 16 100 10

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
        if(inputPoints.count() < K){
            System.out.println("WARNING: Input file is smaller than K number of clusters!");
        }

        // as requested, we count the sizes of the two groups
        long N = inputPoints.count();
        long NA = inputPoints.filter(row -> row.endsWith("A")).count();
        long NB = inputPoints.filter(row -> row.endsWith("B")).count();
        System.out.println("N = " + N + ", NA= " + NA + ", NB = " + NB);

        // as requested, we first convert the strings given as input into Vector points
        JavaRDD<Vector> parsedInputPoints = methodsHW2.conversion(inputPoints);

        long start = System.currentTimeMillis();
        KMeansModel clusters = KMeans.train(parsedInputPoints.rdd(), K, M);
        Vector[] C_stand = clusters.clusterCenters();
        long end = System.currentTimeMillis();
        long time_C_stand = end-start;

        // IMPORTANT NOTE: The clusters found in C_stand could have different indices than C_fair even if they are the same
        start = System.currentTimeMillis();
        Vector[] C_fair = methodsHW2.MRFairLloyd(inputPoints, K, M);
        end = System.currentTimeMillis();
        long time_C_fair = end-start;

        start = System.currentTimeMillis();
        double phi_stand = methodsHW2.MRComputeFairObjective(inputPoints, C_stand);
        end = System.currentTimeMillis();
        long time_phi_stand = end-start;
        System.out.println("Fair Objective with Standard Centers = " + String.format("%.4f", phi_stand));
        start = System.currentTimeMillis();
        double phi_fair = methodsHW2.MRComputeFairObjective(inputPoints, C_fair);
        end = System.currentTimeMillis();
        long time_phi_fair = end-start;
        System.out.println("Fair Objective with Fair Centers = " + String.format("%.4f", phi_fair));
        System.out.println("Time to compute standard centers = " + time_C_stand + " ms");
        System.out.println("Time to compute fair centers = " + time_C_fair + " ms");
        System.out.println("Time to compute objective with standard centers = " + time_phi_stand + " ms");
        System.out.println("Time to compute objective with fair centers = " + time_phi_fair + " ms");
    }
}

class methodsHW2{
    public static Vector[] MRFairLloyd(JavaRDD<String> inputPoints, int K, int M){
        JavaRDD<Tuple2<Vector, String>> parsedInputPoints = conversionWithGroups(inputPoints);
        int d = getPointsDimensionality(parsedInputPoints);
        long countInputPointsA = parsedInputPoints.filter(point -> point._2().endsWith("A")).count(); // | A |
        long countInputPointsB = parsedInputPoints.filter(point -> point._2().endsWith("B")).count(); // | B |

        // M iterations of Fair K-Means Clustering
        KMeansModel model = KMeans.train(parsedInputPoints.map(Tuple2::_1).rdd(), K, 0);
        Vector[] centers = model.clusterCenters();

        for(int i = 0; i < M; i++){
            // applying Lloyd to associate the clusters with their centers
            // in this case, each cluster is a pair indexOfTheCluster-ListOfTuples, where each tuple is a point and its corresponding group

            model = new KMeansModel(centers);

            KMeansModel finalModel = model;
            JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> pairsPointsToCluster = parsedInputPoints.mapToPair(point -> {
                return new Tuple2<>(finalModel.predict(point._1()), point); // returns a pair point-index of the corresponding cluster
            }).groupByKey(); // groups points of the same cluster

            JavaPairRDD<Integer, ArrayList<Tuple2<Vector, String>>> clustersGroupA = getClustersOfGroup(pairsPointsToCluster, "A").cache(); // cluster A
            JavaPairRDD<Integer, ArrayList<Tuple2<Vector, String>>> clustersGroupB = getClustersOfGroup(pairsPointsToCluster, "B").cache(); // cluster B

            // HW2 assumes K small enough that all static arrays below can be stored as local variables instead of RDD's.
            long[] clustersCountA = getClustersGroupCounts(clustersGroupA, K); // | A ∩ U_i |
            long[] clustersCountB = getClustersGroupCounts(clustersGroupB, K); // | B ∩ U_i |

            // α_i and β_i
            double[] alphas = getGroupPartialCounts(clustersCountA, countInputPointsA);
            double[] betas = getGroupPartialCounts(clustersCountB, countInputPointsB);

            // M^A and M^B
            Tuple2<Vector[], Vector[]> groupCentroids = getGroupsCentroids(clustersGroupA, clustersGroupB, clustersCountA, clustersCountB, K);
            Vector[] groupCentroidA = groupCentroids._1();
            Vector[] groupCentroidB = groupCentroids._2();

            // ell
            double[] euclNorms = getEuclideanNorms(groupCentroidA, groupCentroidB, K);

            // prepare parameters for computeVectorX, which is used as-is from the HW
            double fixedA = getTotalDistanceForEachCluster(clustersGroupA, groupCentroidA)/countInputPointsA;
            double fixedB = getTotalDistanceForEachCluster(clustersGroupB, groupCentroidB)/countInputPointsB;

            double[] x = computeVectorX(fixedA, fixedB, alphas, betas, euclNorms, K);

            centers = computeFairCenters(euclNorms, x, groupCentroidA, groupCentroidB, d);

            clustersGroupA.unpersist();
            clustersGroupB.unpersist();
        }
        return centers;
    }

    private static double[] getGroupPartialCounts(long[] clustersCountA, long countInputPointsA) {
        double[] result = new double[clustersCountA.length];
        for(int i = 0; i < clustersCountA.length; i++){
            result[i] = (double) clustersCountA[i] /countInputPointsA;
        }
        return result;
    }

    private static Vector[] convertToVectorArray(JavaPairRDD<Integer, Vector> startRDD, int K) {
        Map<Integer, Vector> clusterVectors = startRDD.collectAsMap();
        Vector[] result = new Vector[K];

        clusterVectors.forEach((clusterId, vector) -> result[clusterId] = vector);
        return result;
    }

    private static long[] convertToLongArray(JavaPairRDD<Integer, Long> startRDD, int K) {
        Map<Integer, Long> clusterValues = startRDD.collectAsMap();
        long[] result = new long[K];

        Arrays.fill(result, 690000);
        clusterValues.forEach((clusterId, value) -> result[clusterId] = value);
        return result;
    }


    private static double[] getEuclideanNorms(Vector[] groupCentroidA, Vector[] groupCentroidB, int K) {
        double[] norms = new double[K];
        Arrays.fill(norms, 0.0);

        for (int i = 0; i < K; i++) norms[i] = sqrt(Vectors.sqdist(groupCentroidA[i], groupCentroidB[i]));
        return norms;
    }

    public static JavaPairRDD<Integer, ArrayList<Tuple2<Vector, String>>> getClustersOfGroup(JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> pairsPointsToCluster, String group) {
        return pairsPointsToCluster.mapValues(cluster -> {
            ArrayList<Tuple2<Vector, String>> groupPointsInCluster = new ArrayList<>();
            for(Tuple2<Vector, String> tuple : cluster){
                if(tuple._2().endsWith(group)) groupPointsInCluster.add(tuple);
            }
            return groupPointsInCluster;
        }).sortByKey();
    }

    private static long[] getClustersGroupCounts(JavaPairRDD<Integer, ArrayList<Tuple2<Vector, String>>> clustersGroup, int K) {
        JavaPairRDD<Integer, Long> counts = clustersGroup.mapToPair(pair -> {
            long count = 0L;
            for (Tuple2<Vector, String> tuple : pair._2()) {
                count++;
            }
            return new Tuple2<>(pair._1(), count);
        });
        return convertToLongArray(counts, K);
    }

    private static Tuple2<Vector[], Vector[]> getGroupsCentroids(JavaPairRDD<Integer, ArrayList<Tuple2<Vector, String>>> clustersGroupA, JavaPairRDD<Integer, ArrayList<Tuple2<Vector, String>>> clustersGroupB, long[] clustersGroupCountsA, long[] clustersGroupCountsB, int K){
        Vector[] groupCentroidA = getCentroidsGroupedClusters(clustersGroupA, clustersGroupCountsA, K);
        Vector[] groupCentroidB = getCentroidsGroupedClusters(clustersGroupB, clustersGroupCountsB, K);
        // check size of each vector and set the empty ones to the one of the other group, so that ell is = 0
        for(int i = 0; i < groupCentroidA.length; i++){
            if(groupCentroidA[i].size() == 0){
                groupCentroidA[i] = groupCentroidB[i];
            }
        }
        for(int i = 0; i < groupCentroidB.length; i++){
            if(groupCentroidB[i].size() == 0){
                groupCentroidB[i] = groupCentroidA[i];
            }
        }
        return new Tuple2<>(groupCentroidA, groupCentroidB);
    }

    private static Vector[] getCentroidsGroupedClusters(JavaPairRDD<Integer, ArrayList<Tuple2<Vector, String>>> clustersGroup, long[] clustersGroupCounts, int K) {
        JavaPairRDD<Integer, Vector> clustersMeans = clustersGroup.mapToPair(clusterPair -> {
            double[] sum = {};
            for (Tuple2<Vector, String> point : clusterPair._2()) {
                double[] pointComponents = point._1().toArray();
                if(sum.length == 0){
                    sum = new double[pointComponents.length];
                    Arrays.fill(sum, 0.0);
                }
                // sum each component of each element and save the summed components in sum
                for(int i = 0; i < pointComponents.length; i++) sum[i] += pointComponents[i];
            }
            double[] centroid = new double[sum.length];
            int clusterId = clusterPair._1();
            // compute the centroid for each component of the sum vector
            for(int i = 0; i < sum.length; i++){
                centroid[i] = sum[i] / clustersGroupCounts[clusterId];
            }
            Vector ithCentroid = Vectors.dense(centroid);
            return new Tuple2<>(clusterId, ithCentroid);
        });
        return convertToVectorArray(clustersMeans, K);
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

    private static int getPointsDimensionality(JavaRDD<Tuple2<Vector, String>> parsedInputPoints){
        return parsedInputPoints.map(row ->{
           Vector point = row._1();
           return point.size();
        }).reduce(Math::max);
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

    private static Vector[] computeFairCenters(double[] ellValues, double[] x, Vector[] groupCentroidA, Vector[] groupCentroidB, int d){
        Vector[] centers = new Vector[groupCentroidA.length];
            // for each cluster i: meanA, meanB are vectors, all the others are scalars. Also, the center is a vector.
            // so for each cluster we first apply the formula for the single components of the vectors, so that we can get a single component of the i-th Center,
        for (int i = 0; i < centers.length; i++) {
            Vector ithCentroidA = groupCentroidA[i];
            Vector ithCentroidB = groupCentroidB[i];

            double[] ithClusterFairCenter = new double[d];
            // We have to do some checks on the addends of the additions to make sure
            // the centroid can be computed even in the case one of the two clusters is 0
            double addendA = 0.0;
            double addendB = 0.0;
            // for each component of the ithCentroid's compute each component of the centers
            for(int j = 0; j < d; j++){
                if(ellValues[i] > 0.0){
                    addendA = (ellValues[i] - x[i]) * ithCentroidA.apply(j);
                    addendB = x[i] * ithCentroidB.apply(j);
                    ithClusterFairCenter[j] = (addendA+addendB) / ellValues[i];
                }
                else ithClusterFairCenter[j] = ithCentroidA.apply(j); // case "one empty cluster"
            }
            // than we gather all the components of the i-th Center into one Vector of doubles
            centers[i] = Vectors.dense(ithClusterFairCenter);
        }
        return centers;
    }

    // this method was designed in HW1, but we actually needed the totalDistance for HW2, so we extracted that part into its own method
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

    private static double getTotalDistanceForEachCluster(JavaPairRDD<Integer, ArrayList<Tuple2<Vector, String>>> clustersGroup, Vector[] means) {
        double totalDistance = clustersGroup.map(pair -> {
            int clusterId = pair._1();
            ArrayList<Tuple2<Vector, String>> cluster = pair._2();
            Vector ithCentroid = means[clusterId];

            double ithTotDistanceFromClusters = 0;
            for (Tuple2<Vector, String> point : cluster) {
                ithTotDistanceFromClusters += Vectors.sqdist(point._1(), ithCentroid);
            }
            return ithTotDistanceFromClusters;
        }).reduce(Double::sum);
        return totalDistance;
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
