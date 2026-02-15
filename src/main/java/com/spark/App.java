package com.spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import static org.apache.spark.sql.functions.*;

public class App {

    // Constant inside the class, but outside methods
    public static final String FILE_PATH = "src/main/resources/IMDB.csv";

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("IMDB_Analytics")
                .master("local[*]")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        // Load the dataset
        // multiLine + escape handle descriptions containing double-double quotes (e.g. ""word"")
        // that otherwise cause CSV column misalignment for ~257 rows
        Dataset<Row> rawData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("multiLine", "true")
                .option("escape", "\"")
                .csv(FILE_PATH);

        // Task 1: Cleaning and Preprocessing
        Dataset<Row> cleanedData = performCleaning(rawData);
        cleanedData.cache(); // Cache — branching point for most tasks
        cleanedData.show();

        // Stage 3: Exploded genres — shared by Tasks 2 and 6
        // Split comma-separated genres into individual rows, filter out "Unknown"
        Dataset<Row> explodedGenres = cleanedData
                .filter(col("rating").isNotNull()) // Need ratings for genre-based analysis
                .withColumn("genre", explode(split(col("genre"), ",\\s*")))
                .filter(not(col("genre").equalTo("Unknown")))
                .cache();

        // Task 2: Top Rated Movies by Genre
        Dataset<Row> topRatedByGenre = getTopRatedByGenre(explodedGenres);
        topRatedByGenre.show(100, false); // Show more rows to see multiple genres

        spark.stop();
    } // End of the main method

    /**
     * Task 1: Data Cleaning and Preprocessing
     *
     * Strategy:
     * - Fill string columns (certificate, duration, genre) with "Unknown" to preserve rows.
     * - Keep rating and votes as null where missing. Each downstream task will
     *   filter out nulls on the columns it actually needs, avoiding unnecessary
     *   data loss for tasks that don't require numeric fields (e.g. Task 3 only
     *   needs stars, Task 5 only needs titles).
     * - Convert votes from comma-formatted string to integer where present.
     * - Derive a "type" column (TV Show vs Movie) from the year column BEFORE
     *   extracting the numeric year, since the range pattern (e.g. "2015–2022")
     *   is what distinguishes TV shows from movies.
     * - Extract the first numeric year value for consistency.
     */
    private static Dataset<Row> performCleaning(Dataset<Row> df) {
        return df
                // Fill missing string columns with "Unknown"
                .na().fill("Unknown", new String[]{"certificate", "duration", "genre"})

                // Clean votes: remove commas, then only cast values that are purely numeric.
                // Empty strings or other non-numeric values become null.
                .withColumn("votes", regexp_replace(col("votes"), ",", ""))
                .withColumn("votes",
                        when(col("votes").rlike("^\\d+$"), col("votes").cast("int"))
                                .otherwise(lit(null).cast("int")))

                // Derive type column BEFORE extracting numeric year
                // TV shows have an en-dash (\u2013) in the year field (e.g. "2015–2022" or "2018– ")
                .withColumn("type",
                        when(col("year").contains("\u2013"), lit("TV Show"))
                                .otherwise(lit("Movie")))

                // Extract the first numeric year value.
                // regexp_extract returns "" when no match; convert to null before casting.
                .withColumn("year", regexp_extract(col("year"), "(\\d+)", 1))
                .withColumn("year",
                        when(col("year").equalTo(""), lit(null).cast("int"))
                                .otherwise(col("year").cast("int")));
    }

    /**
     * Task 2: Top Rated Movies by Genre
     *
     * Uses a window function to rank movies within each genre by rating (descending),
     * with votes as a tiebreaker. This is more efficient than a groupBy + sort approach
     * because Spark's Catalyst optimizer can push the ranking into a single shuffle stage.
     *
     * Input: The exploded genres dataset (one row per movie-genre pair), already
     *        filtered for non-null ratings and non-"Unknown" genres.
     * Output: Top 10 movies per genre, ranked by rating then votes.
     */
    private static Dataset<Row> getTopRatedByGenre(Dataset<Row> explodedGenres) {
        // Define window: partition by genre, order by rating desc then votes desc as tiebreaker
        WindowSpec genreWindow = Window
                .partitionBy("genre")
                .orderBy(col("rating").desc(), col("votes").desc());

        return explodedGenres
                .withColumn("rank", row_number().over(genreWindow))
                .filter(col("rank").leq(10))
                .select("genre", "rank", "title", "rating", "votes", "year")
                .orderBy("genre", "rank");
    }
} // End of class