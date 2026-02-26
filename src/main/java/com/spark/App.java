package com.spark;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.explode;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.lower;
import static org.apache.spark.sql.functions.max;
import static org.apache.spark.sql.functions.min;
import static org.apache.spark.sql.functions.not;
import static org.apache.spark.sql.functions.regexp_extract;
import static org.apache.spark.sql.functions.regexp_replace;
import static org.apache.spark.sql.functions.row_number;
import static org.apache.spark.sql.functions.split;
import static org.apache.spark.sql.functions.stddev;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.trim;
import static org.apache.spark.sql.functions.when;

public class App {

    // Path to the IMDB dataset CSV file
    public static final String FILE_PATH = "src/main/resources/IMDB.csv";

    // Predefined stop words for Task 5
    public static final List<String> STOP_WORDS = Arrays.asList(
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "is", "of", "it"
    );

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

        // =====================================================================
        // CACHING STRATEGY
        // =====================================================================
        // Spark uses lazy evaluation — transformations build a DAG but don't
        // execute until an action (e.g. show(), write()) triggers computation.
        // We cache at DAG branching points where multiple downstream tasks
        // depend on the same intermediate result. This tells Spark to
        // materialize and store the result in memory after the first action,
        // so subsequent tasks reuse it instead of recomputing from the CSV.
        //
        // Cache points:
        //   cleanedData     → used by Tasks 2, 3, 4, 5, 6, 7, 8, 9, 10 (root branch)
        //   explodedGenres  → used by Tasks 2 and 6 (genre-based branch)
        //   certRatings     → used by Tasks 7 and 9 (identical output, saved twice)
        //   validRatedWorks → used by Tasks 8 and 10 (comparison branch)
        // =====================================================================

        // ---------------------------------------------------------
        // Task 1: Cleaning and Preprocessing
        // ---------------------------------------------------------
        Dataset<Row> cleanedData = performCleaning(rawData);
        cleanedData.cache(); // Cache — root branching point for all tasks
        cleanedData.show();
        saveResult(cleanedData, "output/task1_cleaned_data");

        // ---------------------------------------------------------
        // Shared Stage: Exploded genres (used by Tasks 2 and 6)
        // ---------------------------------------------------------
        // Split comma-separated genres into individual rows, filter out "Unknown"
        Dataset<Row> explodedGenres = cleanedData
                .filter(col("rating").isNotNull()) // Need ratings for genre-based analysis
                .withColumn("genre", explode(split(col("genre"), ",\\s*")))
                .filter(not(col("genre").equalTo("Unknown")))
                .cache(); // Cache — branching point for Tasks 2 & 6

        // ---------------------------------------------------------
        // Task 2: Top Rated Movies by Genre
        // ---------------------------------------------------------
        Dataset<Row> topRatedByGenre = getTopRatedByGenre(explodedGenres);
        saveResult(topRatedByGenre, "output/task2_top_rated_by_genre");

        // ---------------------------------------------------------
        // Task 3: Actor Collaboration Network
        // ---------------------------------------------------------
        Dataset<Row> actorCollaboration = getActorCollaboration(cleanedData);
        saveResult(actorCollaboration, "output/task3_actor_collaboration_network");

        // ---------------------------------------------------------
        // Task 4: High-Rated Hidden Gems
        // ---------------------------------------------------------
        Dataset<Row> hiddenGems = getHiddenGems(cleanedData);
        saveResult(hiddenGems, "output/task4_hidden_gems");

        // ---------------------------------------------------------
        // Task 5: Word Frequency in Movie Titles
        // ---------------------------------------------------------
        Dataset<Row> wordFrequencyInTitles = getWordFrequencyInTitles(cleanedData);
        saveResult(wordFrequencyInTitles, "output/task5_word_frequency");

        // ---------------------------------------------------------
        // Task 6: Genre Diversity in Ratings
        // ---------------------------------------------------------
        Dataset<Row> genreDiversity = getGenreDiversity(explodedGenres);
        saveResult(genreDiversity, "output/task6_genre_diversity");

        // ---------------------------------------------------------
        // Tasks 7 & 9: Certification Rating Distribution
        // ---------------------------------------------------------
        // Tasks 7 and 9 have identical objectives, therefore we compute
        // once and cache to avoid redundant computation.
        Dataset<Row> certRatings = getCertificationRating(cleanedData);
        certRatings.cache(); // Cache BEFORE first action
        saveResult(certRatings, "output/task7_certification_rating");
        saveResult(certRatings, "output/task9_certification_rating");

        // ---------------------------------------------------------
        // Shared Stage: Deduplicated works (used by Tasks 8 and 10)
        // ---------------------------------------------------------
        // Filter out nulls and deduplicate by (title, type, year) once,
        // avoiding redundant shuffles across both tasks.
        Dataset<Row> validRatedWorks = getDeduplicatedWorks(cleanedData);
        validRatedWorks.cache(); // Cache — branching point for Tasks 8 & 10

        // ---------------------------------------------------------
        // Task 8: Comparing TV Shows and Movies — Overall Summary
        // ---------------------------------------------------------
        Dataset<Row> overallSummary = getOverallTvVsMovies(validRatedWorks);
        saveResult(overallSummary, "output/task8_tv_vs_movies_summary");

        // ---------------------------------------------------------
        // Task 10: Comparing TV Shows and Movies — Trends Over Time
        // ---------------------------------------------------------
        Dataset<Row> trendsByYear = getTvVsMoviesTrends(validRatedWorks);
        saveResult(trendsByYear, "output/task10_tv_vs_movies_trends");

        spark.stop();
    }

    // =================================================================
    //  TASK METHODS
    // =================================================================

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
     *   extracting the numeric year, since the range pattern (e.g. "2015-2022")
     *   is what distinguishes TV shows from movies.
     * - Improved: Also detects TV Shows if the certificate starts with "TV"
     *   (e.g. "TV-MA"), which catches miniseries that lack a year range.
     * - Extract the first numeric year value for consistency.
     */
    private static Dataset<Row> performCleaning(Dataset<Row> df) {
        return df
                // Fill missing string columns with "Unknown"
                .na().fill("Unknown", new String[]{"certificate", "duration", "genre"})

                // Clean votes: remove commas, cast to int
                .withColumn("votes", regexp_replace(col("votes"), ",", ""))
                .withColumn("votes",
                        when(col("votes").rlike("^\\d+$"), col("votes").cast("int"))
                                .otherwise(lit(null).cast("int")))

                // Robust type detection:
                // A title is classified as a TV Show if:
                //   1. The year contains an en-dash (e.g. "2015-2022"), OR
                //   2. The certificate starts with "TV" (e.g. "TV-MA")
                // This dual check catches miniseries that have a single year
                // but a TV-prefixed certificate.
                .withColumn("type",
                        when(col("year").contains("\u2013")       // \u2013 = en-dash
                                        .or(col("certificate").startsWith("TV")),
                                lit("TV Show"))
                                .otherwise(lit("Movie")))

                // Extract the first numeric year from the string (e.g. "2015" from "(2015-2022)")
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
        Dataset<Row> distinctMovies = getUniqueMoviesPerGenre(explodedGenres);

        WindowSpec genreWindow = Window
                .partitionBy("genre")
                .orderBy(col("rating").desc(), col("votes").desc());

        return distinctMovies
                .withColumn("rank", row_number().over(genreWindow))
                .filter(col("rank").leq(10))
                .select("genre", "rank", "title", "rating", "votes", "year")
                .orderBy("genre", "rank");
    }

    /**
     * Task 3: Actor Collaboration Network
     *
     * Analyzes actor collaborations by creating a co-occurrence network.
     * Parses the stars column to extract individual actor names, generates
     * pairwise combinations for each title, and counts collaborations.
     *
     * The stars column has two formats:
     *   1. Clean:         ['Actor1, ', 'Actor2, ', 'Actor3']
     *   2. With director: ['Director', '| ', '    Stars:', 'Actor1, ', 'Actor2']
     * We strip brackets/quotes, split by comma, trim whitespace, then filter
     * out metadata entries ('|', 'Stars:') to keep only actor names.
     *
     * Deduplication by (title, actor) ensures that multi-episode shows sharing
     * the same title (e.g. 92 "Top Gear" episodes) count as one collaboration
     * per unique title, not one per episode.
     *
     * The self-join with (Actor1 < Actor2) ensures each pair is counted once
     * and self-pairings are excluded.
     *
     * Output: Actor pairs with collaboration count, sorted by most collaborations.
     */
    private static Dataset<Row> getActorCollaboration(Dataset<Row> df) {
        // 1. Parse and clean the stars column
        Dataset<Row> actorMovieMap = df
                .filter(col("stars").isNotNull())
                .withColumn("stars_cleaned", regexp_replace(col("stars"), "[\\[\\]'\"]", ""))
                .withColumn("actor", explode(split(col("stars_cleaned"), ",")))
                .withColumn("actor", trim(col("actor")))
                // Filter out empty strings and metadata entries (not actor names)
                .filter(col("actor").notEqual(""))
                .filter(not(col("actor").equalTo("|")))
                .filter(not(col("actor").startsWith("Star")))
                .filter(not(col("actor").contains("|")))
                .select("title", "actor")
                // Deduplicate: count each actor once per title, so multi-episode
                // shows (e.g. 92 "Top Gear" rows) don't inflate pair counts
                .dropDuplicates("title", "actor");

        // 2. Self-join on title to generate pairwise actor combinations
        // The (a.actor < b.actor) condition ensures each pair appears once
        // and prevents self-pairing.
        Dataset<Row> collaborations = actorMovieMap.as("a")
                .join(actorMovieMap.as("b"), "title")
                .where(col("a.actor").lt(col("b.actor")))
                .select(
                        col("a.actor").as("Actor1"),
                        col("b.actor").as("Actor2")
                );

        // 3. Count collaborations per pair
        return collaborations.groupBy("Actor1", "Actor2")
                .count()
                .withColumnRenamed("count", "collaboration_count")
                .orderBy(col("collaboration_count").desc());
    }

    /**
     * Task 4: High-Rated Hidden Gems
     *
     * Identifies movies with high ratings (> 8.0) but relatively low votes (< 10,000).
     * These are quality titles that haven't received mainstream attention.
     * Filters to movies only and sorts by rating desc, votes desc.
     */
    private static Dataset<Row> getHiddenGems(Dataset<Row> df) {
        return df
                .filter(col("type").equalTo("Movie"))
                .filter(col("rating").gt(8.0))
                .filter(col("votes").lt(10000))
                .orderBy(col("rating").desc(), col("votes").desc());
    }

    /**
     * Task 5: Word Frequency in Movie Titles
     *
     * Performs a word count on the title column, excluding common stop words.
     * Titles are lowercased and split by whitespace, then punctuation is removed
     * BEFORE filtering stop words to ensure accurate matching (e.g. "love," is
     * cleaned to "love" before the stop word check, not after).
     *
     * Output: Top 20 most frequent words in movie titles.
     */
    private static Dataset<Row> getWordFrequencyInTitles(Dataset<Row> df) {
        // 1. Lowercase, split, and explode titles into individual words
        Dataset<Row> wordsDf = df
                .filter(col("title").isNotNull())
                .withColumn("word", explode(split(lower(col("title")), "\\s+")))
                // 2. Clean punctuation BEFORE filtering stop words
                .withColumn("word", regexp_replace(col("word"), "[^a-zA-Z]", ""))
                .filter(col("word").notEqual(""))
                // 3. Remove stop words
                .filter(not(col("word").isin(STOP_WORDS.toArray())));

        // 4. Count word frequencies and return Top 20
        return wordsDf.groupBy("word")
                .count()
                .withColumnRenamed("count", "word_frequency")
                .orderBy(col("word_frequency").desc())
                .limit(20);
    }

    /**
     * Task 6: Genre Diversity in Ratings
     *
     * Measures rating variability across genres using standard deviation.
     * Additional statistics (mean, min, max, range) provide richer context
     * for identifying which genres have the most consistent vs. polarizing ratings.
     *
     * Input: Exploded genres dataset, filtered to unique movies per genre
     *        (via getUniqueMoviesPerGenre) to avoid duplicate titles inflating stats.
     * Output: One row per genre with diversity metrics, sorted by stddev descending.
     *         Genres with fewer than 5 movies are excluded for statistical reliability.
     */
    private static Dataset<Row> getGenreDiversity(Dataset<Row> explodedGenres) {
        Dataset<Row> distinctMovies = getUniqueMoviesPerGenre(explodedGenres);

        return distinctMovies
                .groupBy("genre")
                .agg(
                        count("*").alias("movie_count"),
                        avg("rating").alias("avg_rating"),
                        stddev("rating").alias("rating_stddev"),
                        min("rating").alias("min_rating"),
                        max("rating").alias("max_rating"),
                        expr("max(rating) - min(rating)").alias("rating_range")
                )
                .filter(col("movie_count").gt(5))
                .orderBy(col("rating_stddev").desc());
    }

    /**
     * Tasks 7 & 9: Certification Rating Distribution
     *
     * Analyzes the frequency of different content certifications (e.g., PG-13, R, TV-MA)
     * and calculates the average rating per certification type.
     * Filters out rows with unknown certificates and null ratings for accuracy.
     * Sorted by average rating descending to highlight which certifications
     * tend to be best received by audiences.
     *
     * Input: Cleaned IMDB dataset.
     * Output: One row per certification with movie count and average rating.
     */
    private static Dataset<Row> getCertificationRating(Dataset<Row> df) {
        return df
                .filter(col("rating").isNotNull())
                .filter(not(col("certificate").equalTo("Unknown")))
                .groupBy("certificate")
                .agg(
                        count("certificate").alias("movies_certificate_count"),
                        avg("rating").alias("certificate_avg_rating")
                )
                .orderBy(col("certificate_avg_rating").desc());
    }

    /**
     * Task 8: Comparing TV Shows and Movies — Overall Summary
     *
     * Aggregates across the entire dataset to compare Movies vs TV Shows
     * on average rating, rating spread, average votes per title, and total votes.
     *
     * Input: Pre-filtered and deduplicated dataset (from getDeduplicatedWorks).
     * Output: One row per type with aggregate statistics.
     */
    private static Dataset<Row> getOverallTvVsMovies(Dataset<Row> deduped) {
        return deduped
                .groupBy("type")
                .agg(
                        count("*").alias("count"),
                        avg("rating").alias("avg_rating"),
                        stddev("rating").alias("stddev_rating"),
                        avg("votes").alias("avg_votes"),
                        sum("votes").alias("total_votes"),
                        min("rating").alias("min_rating"),
                        max("rating").alias("max_rating")
                )
                .orderBy("type");
    }

    /**
     * Task 10: Comparing TV Shows and Movies — Trends Over Time
     *
     * Groups by type and year to reveal how average ratings and total votes
     * evolved over time. Requires non-null year for meaningful trend analysis.
     *
     * Input: Pre-filtered and deduplicated dataset (from getDeduplicatedWorks).
     * Output: One row per (type, year) with aggregate statistics, sorted by
     *         year ascending to show chronological trends.
     */
    private static Dataset<Row> getTvVsMoviesTrends(Dataset<Row> deduped) {
        return deduped
                .filter(col("year").isNotNull())
                .groupBy("type", "year")
                .agg(
                        count("*").alias("count"),
                        avg("rating").alias("avg_rating"),
                        avg("votes").alias("avg_votes"),
                        sum("votes").alias("total_votes")
                )
                .orderBy(col("year").asc(), col("type"));
    }

    // =================================================================
    //  SHARED HELPERS
    // =================================================================

    /**
     * Filters for Movies and deduplicates by Title per Genre.
     * Used by Tasks 2 and 6 to ensure consistent, clean data.
     *
     * Some titles appear multiple times in the dataset (e.g. different episodes
     * or re-releases). We keep only the highest-rated entry per title within
     * each genre, using votes as a tiebreaker.
     */
    private static Dataset<Row> getUniqueMoviesPerGenre(Dataset<Row> explodedGenres) {
        Dataset<Row> moviesOnly = explodedGenres
                .filter(col("type").equalTo("Movie"));

        WindowSpec titleDedupeWindow = Window
                .partitionBy("genre", "title")
                .orderBy(col("rating").desc(), col("votes").desc());

        return moviesOnly
                .withColumn("title_rank", row_number().over(titleDedupeWindow))
                .filter(col("title_rank").equalTo(1))
                .drop("title_rank");
    }

    /**
     * Filters and deduplicates works for TV vs Movie comparison.
     * Used by Tasks 8 and 10 to avoid repeating the same filter + window shuffle.
     *
     * - Requires non-null rating AND votes for meaningful aggregation.
     * - Deduplicates by (title, type, year), keeping the highest-rated entry.
     *   This prevents inflated vote totals from duplicate rows (e.g. "Vagabond"
     *   appearing twice) while preserving distinct works that share a title
     *   (e.g. a Movie and a TV Show with the same name).
     */
    private static Dataset<Row> getDeduplicatedWorks(Dataset<Row> df) {
        Dataset<Row> valid = df.filter(
                col("rating").isNotNull().and(col("votes").isNotNull()));

        WindowSpec dedupWindow = Window
                .partitionBy("title", "type", "year")
                .orderBy(col("rating").desc(), col("votes").desc());

        return valid
                .withColumn("rank", row_number().over(dedupWindow))
                .filter(col("rank").equalTo(1))
                .drop("rank");
    }

    /**
     * Saves a Dataset to a single CSV file.
     * coalesce(1) merges all partitions into one file for easy inspection.
     */
    private static void saveResult(Dataset<Row> df, String outputPath) {
        df.coalesce(1)
                .write()
                .option("header", "true")
                .mode("overwrite")
                .csv(outputPath);

        System.out.println("Saved results to: " + outputPath);
    }

}