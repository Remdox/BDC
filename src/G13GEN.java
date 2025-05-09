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

public class G13GEN {
    public static void main(String[] args) {
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: N K");
        }
        int N = Integer.parseInt(args[0]);
        int K = Integer.parseInt(args[1]);

        Random rand = new Random(42);

        for(int j=0; j<N; j++) {

        }
    }
}
