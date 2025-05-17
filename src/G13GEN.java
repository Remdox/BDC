import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class G13GEN {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) throw new IllegalArgumentException("USAGE: N K");
        int N = Integer.parseInt(args[0]);
        int K = Integer.parseInt(args[1]);

        Random rand = new Random();
        int[][] directions = {{1,1}, {1,-1}, {-1,-1}, {-1,1}};
        double clusterDistance = 1000.0;
        double clusterSpread = 5.0;
        double shiftPointsB = 100.0;

        try (PrintWriter outFile = new PrintWriter(new FileWriter("unfairDataset.csv"))) {
            for (int k = 0; k < K; k++) {
                int dirIdx = k % 4;
                int dirX = directions[dirIdx][0];
                int dirY = directions[dirIdx][1];

                int clusterSize = N / K;
                double freqB = 0.1;
                int pointsB = Math.max(1, (int) Math.round(clusterSize * freqB));
                int pointsA = clusterSize - pointsB;

                double centroidX = dirX * (clusterDistance + (k/4)*100000);
                double centroidY = dirY * (clusterDistance + (k/4)*100000);

                for (int j = 0; j < pointsA; j++) {
                    double x = centroidX + rand.nextGaussian() * clusterSpread;
                    double y = centroidY + rand.nextGaussian() * clusterSpread;
                    outFile.printf(Locale.US, "%.3f,%.3f,A%n", x, y);
                }

                double centroidBX = centroidX + dirX * shiftPointsB;
                double centroidBY = centroidY + dirY * shiftPointsB;
                for (int j = 0; j < pointsB; j++) {
                    double x = centroidBX + rand.nextGaussian() * clusterSpread;
                    double y = centroidBY + rand.nextGaussian() * clusterSpread;
                    outFile.printf(Locale.US, "%.3f,%.3f,B%n", x, y);
                }
            }
        }
    }
}