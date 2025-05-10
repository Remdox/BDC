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

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.lang.*;

public class G13GEN {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: N K");
        }
        int N = Integer.parseInt(args[0]);
        int K = Integer.parseInt(args[1]);

        Random rand = new Random();
        int clusterSize = (N/K);
        double frequencyGroupB = 0.09;
        double frequencyGroupA = 1 - frequencyGroupB;
        long clustersDistance = 1000L;
        int multiplier = 10;
        Vector[] points = new Vector[N];

        // create k blobs with same proportions of A,B points
        // for each cluster
        FileWriter fw = new FileWriter("unfairDataset.csv");
        PrintWriter outFile = new PrintWriter(fw);

        // generate clusters for each group
        // for each cluster
        int[][] direction = new int[][]{{1,1}, {1,-1}, {-1, -1}, {-1, 1}};
        int pointsAForCluster = (int) Math.round(clusterSize*frequencyGroupA);
        int pointsBForCluster = clusterSize - pointsAForCluster;
        if(pointsAForCluster == 0) System.out.println("WARNING: ill-defined cluster size or frequency for points of group B. This will result in clusters without points belonging to group B.");
        if(pointsBForCluster == 0) System.out.println("WARNING: ill-defined cluster size or frequency for points of group B. This will result in clusters without points belonging to group B.");
        for(int k = 0; k < K; k++) {
            double[] farthestPosition = new double[]{0.0, 0.0};
            // for each point in the cluster
            // determine each component of the points for group A
            double[] position = new double[]{0.0, 0.0};
            for(int j = 0; j < pointsAForCluster; j++){
                int directionIndex = k % 4;
                position[0] = direction[directionIndex][0] * (clustersDistance + rand.nextGaussian() * 100);
                position[1] = direction[directionIndex][1] * (clustersDistance + rand.nextGaussian() * 100);
                if(j == 0){
                    farthestPosition[0] = position[0];
                    farthestPosition[1] = position[1];
                }
                else{
                    if(Math.abs(position[0]) > Math.abs(farthestPosition[0])) farthestPosition[0] = position[0];
                    if(Math.abs(position[1]) > Math.abs(farthestPosition[1])) farthestPosition[1] = position[1];
                }
                outFile.printf(Locale.US, "%.3f,%.3f,A%n", position[0], position[1]);
            }
            if(k != 0 && (k+1) % 4 == 0) clustersDistance *= 10;
        }
        outFile.close();
    }
}
