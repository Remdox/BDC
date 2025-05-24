import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class G13GEN {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) throw new IllegalArgumentException("USAGE: N K");
        int N = Integer.parseInt(args[0]);
        int K = Integer.parseInt(args[1]);
        if (K > N) System.out.println("WARNING: number of clusters higher than number of points");

        Random rand = new Random();
        int[][] directions = {{1,1}, {1,-1}, {-1,-1}, {-1,1}};
        double clusterDistance = 1000.0;
        double clusterSpread = 5.0;
        double shiftPointsB = 1000.0 + clusterDistance / 20;
        try (PrintWriter outFile = new PrintWriter(new FileWriter("unfairDataset.csv"))) {
            for (int k = 0; k < K; k++) {
                int directionIndex = k % 4;
                int dirX = directions[directionIndex][0];
                int dirY = directions[directionIndex][1];

                int clusterSize = N / K;
                double frequencyGroupB = 0.1;
                int pointsBForCluster = Math.max(1, (int) Math.round(clusterSize * frequencyGroupB));
                int pointsAForCluster = clusterSize - pointsBForCluster;

                if(pointsAForCluster == 0) System.out.println("WARNING: ill-defined cluster size or frequency for points of group A. This will result in clusters without points belonging to group B.");

                double clusterDistanceIncrease = (k/4) * 100000;
                double centroidX = dirX * (clusterDistance + clusterDistanceIncrease);
                double centroidY = dirY * (clusterDistance + clusterDistanceIncrease);

                for (int j = 0; j < pointsAForCluster; j++) {
                    double x = centroidX + rand.nextGaussian() * clusterSpread;
                    double y = centroidY + rand.nextGaussian() * clusterSpread;
                    outFile.printf(Locale.US, "%.3f,%.3f,A%n", x, y);
                }

                double centroidBX = centroidX + dirX * shiftPointsB;
                double centroidBY = centroidY + dirY * shiftPointsB;
                for (int j = 0; j < pointsBForCluster; j++) {
                    double x = centroidBX + rand.nextGaussian() * clusterSpread;
                    double y = centroidBY + rand.nextGaussian() * clusterSpread;
                    outFile.printf(Locale.US, "%.3f,%.3f,B%n", x, y);
                }
            }
        }
    }
}