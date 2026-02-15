package com.spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.regexp_extract;
import static org.apache.spark.sql.functions.regexp_replace;

public class App {

    // Constant inside the class, but outside methods
    public static final String FILE_PATH = "src/main/resources/imdb.csv";

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("IMDB_Analytics")
                .master("local[*]")
                .getOrCreate();

        // Load the dataset [cite: 31]
        Dataset<Row> rawData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(FILE_PATH);

        // Task 1: Cleaning [cite: 34]
        Dataset<Row> cleanedData = performCleaning(rawData);
        cleanedData.show();

        spark.stop();
    } // End of the main method

    // Task 1 Implementation [cite: 35, 36]
    private static Dataset<Row> performCleaning(Dataset<Row> df) {
        return df.na().drop() // Handle missing values [cite: 37]
                // Remove commas from votes [cite: 38]
                .withColumn("votes", regexp_replace(col("votes"), ",", "").cast("int"))
                // Extract numeric year [cite: 39]
                .withColumn("year", regexp_extract(col("year"), "(\\d+)", 1).cast("int"));
    }
} // End of class