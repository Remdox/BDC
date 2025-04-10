import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class WordCountExample{

  public static void main(String[] args) throws IOException {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // CHECKING NUMBER OF CMD LINE PARAMETERS
    // Parameters are: num_partitions, <path_to_file>
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    if (args.length != 2) {
      throw new IllegalArgumentException("USAGE: num_partitions file_path");
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // SPARK SETUP
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);
    SparkConf conf = new SparkConf(true).setAppName("WordCount");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("OFF");

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // INPUT READING
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    // Read number of partitions
    int L = Integer.parseInt(args[0]);

    // Read input file and subdivide it into L random partitions
    JavaRDD<String> docs = sc.textFile(args[1]).repartition(L).cache();

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // SETTING GLOBAL VARIABLES
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    long numdocs, numwords;
    numdocs = docs.count();
    System.out.println("Number of documents = " + numdocs);
    JavaPairRDD<String, Long> wordCounts;
    Random randomGenerator = new Random();

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // 1-ROUND WORD COUNT 
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    wordCounts = docs.flatMapToPair(myMethods::wordCountPerDoc) // <-- MAP PHASE (R1)
            .reduceByKey((x, y) -> x+y);    // <-- REDUCE PHASE (R1)
    numwords = wordCounts.count();
    System.out.println("Number of distinct words in the documents = " + numwords);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // 2-ROUND WORD COUNT - RANDOM KEYS ASSIGNED IN MAP PHASE
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    wordCounts = docs
            .flatMapToPair((document) -> {    // <-- MAP PHASE (R1) () -> { è una lambda function
                String[] tokens = document.split(" ");
                HashMap<String, Long> counts = new HashMap<>();
                ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();
                for (String token : tokens) {
                    counts.put(token, 1L + counts.getOrDefault(token, 0L)); // crea le coppie (parola, frequenzaInDoc_i)
                }
                for (Map.Entry<String, Long> e : counts.entrySet()) {
                    pairs.add(new Tuple2<>(randomGenerator.nextInt(L), new Tuple2<>(e.getKey(), e.getValue()))); // crea le coppie chiave-valore (numero_casuale, (parola, frequenzaInDoc_i)) dove numero_casuale rappresenta una partizione casuale
                }
                return pairs.iterator();
            })
            .groupByKey()    // <-- REDUCE PHASE (R1)  aggrega le coppie con stessa chiave/partizione_casuale, producendo (partizione_casuale, (parola1/2/.., frequenzaInDoc_i/j/..), (parola1/2/.., frequenzaInDoc_k/i/..))
            .flatMapToPair(myMethods::gatherPairs) // <-- REDUCE PHASE (R1) produce {parola, freqInPartiz_J}
            .reduceByKey((x, y) -> x+y); // <-- REDUCE PHASE (R2) aggrega le frequenze di (parola, freqInPartiz_j) nella stessa chiave: {parola, freqInPartiz1, freqInPartiz2...} e poi somma le frequenze producendo {parola, freqTotale}
    numwords = wordCounts.count();
    System.out.println("Number of distinct words in the documents = " + numwords);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // 2-ROUND WORD COUNT - RANDOM KEYS ASSIGNED ON THE FLY
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    wordCounts = docs.flatMapToPair(myMethods::wordCountPerDoc) // per ciascun documento i, wordCountPerDoc produce le coppie (parola-freqInDoc_i)
            .groupBy((wordcountpair) -> randomGenerator.nextInt(L))  // <-- KEY ASSIGNMENT+SHUFFLE+GROUPING --> produce (partizione_casuale, (parola1/2/.., frequenzaInDoc_i/j/..), (parola1/2/.., frequenzaInDoc_k/i/..))
            .flatMapToPair(myMethods::gatherPairs) // produce {parola, freqInPartiz_J}
            .reduceByKey((x, y) -> x+y); // <-- REDUCE PHASE (R2) aggrega le frequenze di (parola, freqInPartiz_j) nella stessa chiave: {parola, freqInPartiz1, freqInPartiz2...} e poi somma le frequenze producendo {parola, freqTotale}
    numwords = wordCounts.count();
    System.out.println("Number of distinct words in the documents = " + numwords);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // 2-ROUND WORD COUNT - SPARK PARTITIONS
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


    wordCounts = docs.flatMapToPair(myMethods::wordCountPerDoc) // per ciascun documento i, wordCountPerDoc produce le coppie parola-frequenzaNelDocumento_i
            .mapPartitionsToPair((element) -> {    // <-- REDUCE PHASE (R1) element è un iteratore alle partizioni create automaticamente da Spark
                 HashMap<String, Long> counts = new HashMap<>();
                 while (element.hasNext()){
                      Tuple2<String, Long> tuple = element.next();
                      counts.put(tuple._1(), tuple._2() + counts.getOrDefault(tuple._1(), 0L)); // in sostanza aggrega tutte le frequenze della parola i dei documenti nella stessa partizione
                 }
                 ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                 for (Map.Entry<String, Long> e : counts.entrySet()) {
                     pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                 }
                 return pairs.iterator(); // restituisce un iteratore alla lista di coppie chiave-valore {parola, occorrenzeNellaPartizione_j}
            })
            .groupByKey()     // <-- SHUFFLE+GROUPING | SEE BELOW, I CAN DO BOTH WITH REDUCEBYKEY
            .mapValues((it) -> { // <-- REDUCE PHASE (R2)
                 long sum = 0;
                 for (long c : it) {
                     sum += c;
                 }
                 return sum;
            }); // Obs: one could use reduceByKey in place of groupByKey and mapValues <----------------- !!!!!!!!
      /*
      SENZA 2-ROUND
      INPUT                                                                                                                            OUTPUT
      (documento, parole) ----MAP-----> (parola,frequenzaNelDoc_i) ----Shuffle---> (parola, freqDoc1, freqDoc2...) ----REDUCE---> (parola, sommaFreq)
                                                                    |________________________________________________________________________________|
                                                                                                    ReduceByKey
      CON 2-ROUND
      Fai prima uno step intermedio di shuffle1 dentro ciascuna partizione (e sommi con REDUCE1), poi fai lo shuffle2 finale con tutte le partizioni.
      Quindi in sostanza aggreghi i dati in 2 step successivi invece che tutto in un unico step, per risparmiare memoria locale e aggiungendo tempo di eleaborazione.
      (documento, parole) ----MAP-----> (parola,frequenzaNelDoc_i) ----PARTIZ+SHUFFLE---> (parola, partiz_i_freqInDoc1, partiz_i_freqInDoc2 ...) ----REDUCE---> (parola, FreqInPartiz_i)------>XX EMPTY MAP -------->   ----Shuffle---> (parola, freqDoc1, freqDoc2...) ----REDUCE---> (parola, sommaFreq)
                                                                      |_________________________________________________________________________________________|                                                       |________________________________________________________________________________|
                                                                                                   mapPartitionsToPair + lambda function                                                                                                                        ReduceByKey
       */
    numwords = wordCounts.count();
    System.out.println("Number of distinct words in the documents = " + numwords);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // COMPUTE AVERAGE WORD LENGTH
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    int avgwordlength = wordCounts
            .map((tuple) -> tuple._1().length())
            .reduce((x, y) -> x+y);
    System.out.println("Average word length = " + (double) avgwordlength/ (double) numwords);

  }

}

class myMethods {
    public static Iterator<Tuple2<String,Long>> wordCountPerDoc(String document) {
        String[] tokens = document.split(" ");
        HashMap<String, Long> counts = new HashMap<>();
        ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
        for (String token : tokens) {
            counts.put(token, 1L + counts.getOrDefault(token, 0L)); // per ciascuna parola del documento, conta quante occorrenze di quella parola ci sono nel documento e salvala nella hashmap
        }
        for (Map.Entry<String, Long> e : counts.entrySet()) { // ora mi serve che la funzione restituisca un iteratore ad una lista di coppie chiave-valore parola-frequenza per il documento i-esimo
            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
        }
        return pairs.iterator(); // questo iteratore all'ArrayList pairs fa anche sì che pairs non si cancelli quando la funzione termina, perchè il garbage collector vede ancora questo riferimento, che infatti viene usato al di fuori della funzione
    } // Quindi pairs verrà cancellato poi dal garbage collector solo DOPO che non verrà più utilizzato al di fuori della funzione

    public static Iterator<Tuple2<String,Long>> gatherPairs(Tuple2<Integer,Iterable<Tuple2<String, Long>>> element) { // element: per ciascuna partizione casuale
        HashMap<String, Long> counts = new HashMap<>();
        for (Tuple2<String, Long> c : element._2()) { // spacchetta [partizione_casuale, ...]
            counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L)); // in sostanza aggrega tutte le frequenze della parola i dei documenti nella stessa partizione
        }
        ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
        for (Map.Entry<String, Long> e : counts.entrySet()) {
            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
        }
        return pairs.iterator(); // produce  {parola, occorrenzeNellaPartizione_j}
    }
}
