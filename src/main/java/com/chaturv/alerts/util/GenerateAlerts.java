package com.chaturv.alerts.util;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Vineet on 15/5/2016.
 */
public class GenerateAlerts {

    public static void main(String[] args) throws IOException {
//        generateDistinctClusters();
        generateOverlappingClusters();
    }

    private static void generateOverlappingClusters() throws IOException {
        Random random = new Random(123L);

        StringBuilder sb = new StringBuilder();
        FileWriter fw = new FileWriter("C:\\work\\repo\\dl4j-0.4-examples\\src\\main\\resources\\alerts_overlapping_clusters.csv");


        for (int i = 0; i < 60; i++) {
            double notional = random.nextFloat() * 1000000 + 2000000; //1m - 3m
            double price = random.nextFloat() * 1.0 + 1.0; //1.0 - 2.0
            sb.append(notional).append(",").append(price).append(",").append(0).append("\n"); //0 class

        }

        for (int i = 0; i < 60; i++) {
            double notional = random.nextFloat() * 3000000 + 2000000; //2m - 5m
            double price = random.nextFloat() * 1.5 + 1.5; //1.5 - 3.0
            sb.append(notional).append(",").append(price).append(",").append(1).append("\n"); //0 class
        }

        for (int i = 0; i < 60; i++) {
            double notional = random.nextFloat() * 3000000 + 3000000; //3m - 6m
            double price = random.nextFloat() * 1.5 + 2.5; //2.5 - 4.0
            sb.append(notional).append(",").append(price).append(",").append(2).append("\n"); //0 class
        }


        fw.write(sb.toString());
        fw.close();
    }

    private static void generateDistinctClusters() throws IOException {
        Random random = new Random(123L);

        StringBuilder sb = new StringBuilder();
        FileWriter fw = new FileWriter("C:\\work\\repo\\dl4j-0.4-examples\\src\\main\\resources\\alerts_distinct_clusters.csv");


        for (int i = 0; i < 60; i++) {
            double notional = random.nextFloat() * 1000000 + 1000000; //1m - 2m
            double price = random.nextFloat() * 0.5 + 1.0; //1.0 - 1.5
            sb.append(notional).append(",").append(price).append(",").append(0).append("\n"); //0 class

        }

        for (int i = 0; i < 60; i++) {
            double notional = random.nextFloat() * 3000000 + 2000000; //2m - 5m
            double price = random.nextFloat() * 0.5 + 1.5; //1.5 - 2.0
            sb.append(notional).append(",").append(price).append(",").append(1).append("\n"); //0 class
        }

        for (int i = 0; i < 60; i++) {
            double notional = random.nextFloat() * 5000000 + 5000000; //5m - 10m
            double price = random.nextFloat() * 0.5 + 2.0; //2.5 - 2.5
            sb.append(notional).append(",").append(price).append(",").append(2).append("\n"); //0 class
        }


        fw.write(sb.toString());
        fw.close();
    }
}

