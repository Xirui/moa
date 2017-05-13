package moa.clusterers.newsClusterer;


public class Main {

    static int defaultTopLevelFeatureSpaceSize = 50;
    static int maxTopSize = 75;
    static int minTopSize = 10;
    static int stepTop = 5;

    static int defaultSubClusteringBinSize = 20;
    static int maxBinSize = 30;
    static int minBinSize = 10;
    static int stepSub = 2;

    static int defaultTopNoveltySize = 100;
    static int maxTopNSize = 120;
    static int minTopNSize = 80;

    static int defaultSubNSize = 20;
    static int maxSubNSize = 30;
    static int minSubNsize = 10;

    static double defaultThreshold = 0.8;
    static double maxThreshold = 1.0;
    static double minThreshold = 0.5;
    static double stepTreshold = 0.1;


    /**
     * Simulates a stream from a given arff file containing strings
     * and uses Kmeans clustering to form a 2 level cluster hierarchy
     *
     * @param args
     */
    public static void main(String[] args) {
        parseCommandLine(args);
        String fileName = "bbc_output.arff";

        testTopLevelFeatureSpaceSize(fileName);
        printSeparator();

        testSubClusteringBinSize(fileName);
        printSeparator();

        testTopNoveltySize(fileName);
        printSeparator();

        testSubNoveltySize(fileName);
        printSeparator();

        testTopNoveltyThreshold(fileName);
        printSeparator();

        testSubNoveltyThreshold(fileName);
        printSeparator();
    }

    public static void printSeparator() {
        System.out.println("--------------------------------------------------");
        System.out.println("--------------------------------------------------");
    }

    public static void testSubNoveltyThreshold(String fileName) {
        System.out.println("Changing subNoveltyThreshold from " + minThreshold + " to " + maxThreshold + " with step of " + stepTreshold);
        System.out.println("subNoveltyThreshold, numTopRecluster, numSubRecluster");
        for (double threshold = minThreshold; threshold <= maxThreshold; threshold += stepTreshold) {
            NewsClusterer.subNoveltyThreshold = threshold;
            NewsClusterer.run(fileName);
        }
        NewsClusterer.subNoveltyThreshold = defaultThreshold;
    }

    public static void testTopNoveltyThreshold(String fileName) {
        System.out.println("Changing topNoveltyThreshold from " + minThreshold + " to " + maxThreshold + " with step of " + stepTreshold);
        System.out.println("topNoveltyThreshold, numTopRecluster, numSubRecluster");
        for (double threshold = minThreshold; threshold <= maxThreshold; threshold += stepTreshold) {
            NewsClusterer.topNoveltyThreshold = threshold;
            NewsClusterer.run(fileName);
        }
        NewsClusterer.topNoveltyThreshold = defaultThreshold;
    }

    public static void testSubNoveltySize(String fileName) {
        System.out.println("Changing subNoveltySize from " + minSubNsize + " to " + maxSubNSize + " with step of " + stepSub);
        System.out.println("subNoveltySize, numTopRecluster, numSubRecluster");
        for (int size = minTopNSize; size <= maxTopNSize; size += stepTop) {
            NewsClusterer.subNoveltySize = size;
            NewsClusterer.run(fileName);
        }
        NewsClusterer.subNoveltySize = defaultSubNSize;
    }

    public static void testTopNoveltySize(String fileName) {
        System.out.println("Changing topNoveltySize from " + minTopNSize + " to " + maxTopNSize + " with step of " + stepTop);
        System.out.println("topNoveltySize, numTopRecluster, numSubRecluster");
        for (int size = minTopNSize; size <= maxTopNSize; size += stepTop) {
            NewsClusterer.topNoveltySize = size;
            NewsClusterer.run(fileName);
        }
        NewsClusterer.topNoveltySize = defaultTopNoveltySize;
    }

    public static void testTopLevelFeatureSpaceSize(String fileName) {
        System.out.println("Changing topLevelFeatureSpaceSize from " + minTopSize + " to " + maxTopSize + " with step of " + stepTop);
        System.out.println("topLevelFeatureSpaceSize, numTopRecluster, numSubRecluster");
        for (int size = minTopSize; size <= maxTopSize; size += stepTop) {
            NewsClusterer.topLevelFeatureSpaceSize = size;
            NewsClusterer.run(fileName);
        }
        NewsClusterer.topLevelFeatureSpaceSize = defaultTopLevelFeatureSpaceSize;
    }

    public static void testSubClusteringBinSize(String fileName) {
        System.out.println("Changing subClusteringBinSize from " + minBinSize + " to " + maxBinSize + " with step of " + stepSub);
        System.out.println("subClusteringBinSize, numTopRecluster, numSubRecluster");
        for (int size = maxBinSize; size <= maxBinSize; size += stepSub) {
            NewsClusterer.subClusteringBinSize = size;
            NewsClusterer.run(fileName);
        }
        NewsClusterer.subClusteringBinSize = defaultSubClusteringBinSize;
    }

    //options string
    final static String USAGE = "Required: <fileInput> \n"
            + "Options\n"
            + "-f size -Top level feature space size (Default-50)\n"
            + "-b size -Bin size before sub clustering (Default-20)\n"
            + "-t size -Novelty bin sie before topic reclustering (Default-100)\n"
            + "-s size -Sub topic novelty before reclustering (Default-20)\n"
            + "-n size -threshold for top level novelty detection as proportion of cluster radius(default 0.8)\n"
            + "-c size -threshold for sub topic level novelty detection as proportion of cluster radius(default 0.8)\n"
            + "-q pathForFeatureSpace -if initial feature space should be saved (Default-no)";

    public static void parseCommandLine(String[] args) {
        if (args.length < 1)
            System.out.println(USAGE);
        else {
            try {
                for (int c = 0; c < args.length; c++) {
                    if (args[c].equals("-f")) {
                        NewsClusterer.topLevelFeatureSpaceSize = Integer.parseInt(args[c + 1]);
                    }
                    if (args[c].equals("-b")) {
                        NewsClusterer.subClusteringBinSize = Integer.parseInt(args[c + 1]);
                    }
                    if (args[c].equals("-t")) {
                        NewsClusterer.topNoveltySize = Integer.parseInt(args[c + 1]);
                    }
                    if (args[c].equals("-s")) {
                        NewsClusterer.subNoveltySize = Integer.parseInt(args[c + 1]);
                    }
                    if (args[c].equals("-q")) {
                        NewsClusterer.SAVE_FEATURE_PATH = args[c + 1];
                        NewsClusterer.IS_SAVING_FEATURE_SPACE = true;
                    }
                    if (args[c].equals("-n")) {
                        NewsClusterer.topNoveltyThreshold = Double.parseDouble(args[c + 1]);
                    }
                    if (args[c].equals("-c")) {
                        NewsClusterer.subNoveltyThreshold = Double.parseDouble(args[c + 1]);
                    }
                }
            } catch (Exception e) {
                System.out.println(USAGE);
            }
        }
    }

}
