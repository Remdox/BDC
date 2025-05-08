import org.apache.datasketches.memory.Memory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.orc.OrcProto;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
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

import static java.lang.Math.sqrt;

class methodsHW2{
    public static List<Vector> MRFairLloyd(JavaRDD<String> inputPoints, Integer K, Integer M){
        JavaRDD<Tuple2<Vector, String>> parsedInputPoints = conversionWithGroups(inputPoints);
        Long countInputPointsA = parsedInputPoints.filter(point -> point._2().endsWith("A")).count();
        Long countInputPointsB = parsedInputPoints.filter(point -> point._2().endsWith("B")).count();

        // M iterations of Fair K-Means Clustering
        List<Vector> centers = new ArrayList<>();
        KMeansModel clusters;
        for(int i = 0; i < M; i++){
            System.out.println("Iteration n."+i);
            // applying Lloyd to associate the clusters with their centers
            // in this case, each cluster is a pair indexOfTheCluster-ListOfTuples, where each tuple is a point and its corresponding group
            if(centers.isEmpty()) clusters = KMeans.train(parsedInputPoints.map(Tuple2::_1).rdd(), K, 0);
            else clusters = new KMeansModel(centers);
            KMeansModel finalClusters = clusters;
            JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> pairsPointsToCluster = parsedInputPoints.mapToPair(point -> {
                return new Tuple2<>(finalClusters.predict(point._1()), point); // returns a pair point-index of the corresponding cluster
            }).groupByKey(); // groups points of the same cluster (TODO may be inefficient?)

            /*pairsPointsToCluster.foreach(cluster -> {
                Integer clusterIndex = cluster._1(); // Indice come Integer
                Iterable<Tuple2<Vector, String>> points = cluster._2();
                System.out.println("\nCluster index (Integer): " + clusterIndex);
                // Stampa solo i primi 5 valori del cluster
                int counter = 0;
                for (Tuple2<Vector, String> point : points) {
                    if (counter >= 5) break; // Limita a 5 punti
                    System.out.println("  Punto " + (counter + 1) + ": " + point._1() + " | Etichetta: " + point._2());
                    counter++;
                }
            });*/

            JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> clustersGroupA = getClustersOfGroup(pairsPointsToCluster, "A"); // cluster A
            JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> clustersGroupB = getClustersOfGroup(pairsPointsToCluster, "B"); // cluster B

            /*clustersGroupA.foreach(cluster -> {
                Integer clusterIndex = cluster._1(); // Indice come Integer
                Iterable<Tuple2<Vector, String>> points = cluster._2();
                System.out.println("\nCluster index (Integer): " + clusterIndex);
                // Stampa solo i primi 5 valori del cluster
                int counter = 0;
                for (Tuple2<Vector, String> point : points) {
                    if (counter >= 5) break; // Limita a 5 punti
                    System.out.println("  Punto " + (counter + 1) + ": " + point._1() + " | Etichetta: " + point._2());
                    counter++;
                }
            });*/

            JavaPairRDD<Integer, Double> clustersCountA = getClustersGroupCounts(clustersGroupA); // A ∩ U_i
            JavaPairRDD<Integer, Double> clustersCountB = getClustersGroupCounts(clustersGroupB); // B ∩ U_i

            // α_i and β_i
            JavaPairRDD<Integer, Double> alphas = clustersCountA.mapToPair(item -> new Tuple2<>(item._1(), item._2() / countInputPointsA));
            JavaPairRDD<Integer, Double> betas = clustersCountB.mapToPair(item -> new Tuple2<>(item._1(), item._2() / countInputPointsB));
            /*betas.take(5).forEach(beta ->
                    System.out.println("Beta - Cluster: " + beta._1() + " | Valore: " + beta._2())
            );*/

            // μ_i
            JavaPairRDD<Integer, Vector> groupMeansA = getMeansClusterGroup(clustersGroupA, countInputPointsA);
            JavaPairRDD<Integer, Vector> groupMeansB = getMeansClusterGroup(clustersGroupB, countInputPointsB);

            // l_i/ell
            JavaPairRDD<Integer, Double> euclNorm = getEuclideanNorm(groupMeansA, groupMeansB);

            // prepare parameters for computeVectorX, which is used as-is from the HW
            // HW2 assumes K small enough that α_i, β_i, l_i can be stored as local variables instead of RDD's.
            double fixedA = getTotalDistanceForEachCluster(clustersGroupA, groupMeansA);
            double fixedB = getTotalDistanceForEachCluster(clustersGroupB, groupMeansB);

            double[] betasValues = convertToArray(betas);
            double[] alphasValues = convertToArray(alphas);
            double[] ellValues = convertToArray(euclNorm);
            for(int k=0; k < ellValues.length; k++){
                System.out.println(ellValues[k]);
            }

            double[] x = computeVectorX(fixedA, fixedB, alphasValues, betasValues, ellValues, K);

            JavaRDD<Vector> centersRDD = computeFairCenters(ellValues, x, groupMeansA, groupMeansB);
            centers = centersRDD.collect();
        }
        return centers;
    }

    private static double[] convertToArray(JavaPairRDD<Integer, Double> startRDD) {
        JavaDoubleRDD valuesRDD = startRDD.mapToDouble(Tuple2::_2);
        List<Double> valuesList = valuesRDD.collect();

        double[] destArray = new double[valuesList.size()];
        for (int i = 0; i < valuesList.size(); i++) destArray[i] = valuesList.get(i);
        return destArray;
    }

    private static Vector convertToVector(JavaRDD<Vector> centers) {
        List<Vector> centerList = centers.collect();
        if (centerList.isEmpty()) return Vectors.dense(new double[0]);

        int dim = centerList.get(0).size();
        double[] concatenated = new double[centerList.size() * dim];

        for (int i = 0; i < centerList.size(); i++) {
            Vector center = centerList.get(i);
            for (int j = 0; j < dim; j++) {
                concatenated[i * dim + j] = center.apply(j);
            }
        }

        return Vectors.dense(concatenated);
    }


    private static JavaPairRDD<Integer, Double> getEuclideanNorm(JavaPairRDD<Integer, Vector> groupMeansA, JavaPairRDD<Integer, Vector> groupMeansB) {
        return groupMeansA.join(groupMeansB).mapToPair(
          groups -> {
              Vector ithMeanA = groups._2()._1();
              Vector ithMeanB = groups._2()._2();

              Double euclideanNorm = sqrt(Vectors.sqdist(ithMeanA, ithMeanB));
              return new Tuple2<>(groups._1(), euclideanNorm);
          }
        );
    }

    public static JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> getClustersOfGroup(JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> pairsPointsToCluster, String group) {
        return pairsPointsToCluster.mapValues(cluster -> {
            ArrayList<Tuple2<Vector, String>> groupPointsInCluster = new ArrayList<>();
            for(Tuple2<Vector, String> tuple : cluster){
                if(tuple._2().endsWith(group)) groupPointsInCluster.add(tuple);
            }
            return groupPointsInCluster;
        });
    }

    private static JavaPairRDD<Integer, Double> getClustersGroupCounts(JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> clustersGroup) {
        JavaPairRDD<Integer, Double> counts = clustersGroup.mapToPair(pair -> {
            long count = 0L;
            for (Tuple2<Vector, String> tuple : pair._2()) {
                count++;
            }
            return new Tuple2<>(pair._1(), (double) count);
        });
        return counts;
    }

    private static JavaPairRDD<Integer, Vector> getMeansClusterGroup(JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> clustersGroup, Long groupCount) {
        JavaPairRDD<Integer, Vector> clustersMeans = clustersGroup.mapToPair(clusterPair -> {
            double[] sum = {};
            for (Tuple2<Vector, String> cluster : clusterPair._2()) {
                double[] points = cluster._1().toArray();
                if(sum.length == 0) sum = new double[points.length];
                else for(int i = 0; i < points.length; i++) sum[i] += points[i];
            }
            double[] mean = new double[sum.length];
            for(int i = 0; i < sum.length; i++) mean[i] = sum[i] / groupCount;
            Vector ithMean = Vectors.dense(mean);
            return new Tuple2<>(clusterPair._1(), ithMean);
        });
        return clustersMeans;
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

    private static JavaRDD<Vector> computeFairCenters(double[] ellValues, double[] x, JavaPairRDD<Integer, Vector> groupMeansA, JavaPairRDD<Integer, Vector> groupMeansB){
        JavaRDD<Vector> centers = groupMeansA.join(groupMeansB).map(groupsMeans -> {
            int i = groupsMeans._1();
            Vector ithMeanA = groupsMeans._2()._1();
            Vector ithMeanB = groupsMeans._2()._2();

            // for each cluster i: meanA, meanB are vectors, all the others are scalars. Also, the center is a vector.
            // so for each cluster we first apply the formula for the single components of the vectors, so that we can get a single component of the i-th Center,
            double[] ithClusterFairCenter = new double[ithMeanA.size()];
            for(int j = 0; j < ithMeanA.size(); j++){
                ithClusterFairCenter[j] = ((ellValues[i] - x[i])*ithMeanA.apply(j) + x[i]*ithMeanB.apply(j)) / ellValues[i];
            }
            // than we gather all the components of the i-th Center into one Vector of doubles
            return Vectors.dense(ithClusterFairCenter);
        });
        return centers;
    }

    // this method was designed in HW1, but we actually needed the totalDistance for HW2, so we extracted that part into its own method
    public static double MRComputeStandardObjective(JavaRDD<Vector> parsedInputPoints, Vector[] C){
        double totalDistance = getTotalDistanceFromCenter(parsedInputPoints, C);
        return (1.0 / parsedInputPoints.count()) * totalDistance;
    }

    private static double getTotalDistanceFromCenter(JavaRDD<Vector> parsedInputPoints, Vector[] C) {
        double totalDistance = parsedInputPoints.map(point -> {
            double minDistance = Double.MAX_VALUE;
            for (Vector c : C) {
                double distance = Vectors.sqdist(point, c);
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            return minDistance;
        }).reduce(Double::sum);
        return totalDistance;
    }

    private static double getTotalDistanceForEachCluster(JavaPairRDD<Integer, Iterable<Tuple2<Vector, String>>> clustersGroup, JavaPairRDD<Integer, Vector> means) {
        double totalDistance = clustersGroup.join(means).map(pair -> {
            Iterable<Tuple2<Vector, String>> cluster = pair._2()._1();
            Vector ithMean = pair._2()._2();

            double ithTotDistanceFromClusters = 0;
            for (Tuple2<Vector, String> point : cluster) {
                Vector value = point._1();
                ithTotDistanceFromClusters += Vectors.sqdist(value, ithMean);
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

public class G13HW2 {
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
        JavaRDD<Vector> parsedInputPoints = methodsHW2.conversion(inputPoints);

        KMeansModel clusters = KMeans.train(parsedInputPoints.rdd(), K, M);
        Vector[] C_stand = clusters.clusterCenters();

        methodsHW2.MRFairLloyd(inputPoints, K, M);
    }
}
