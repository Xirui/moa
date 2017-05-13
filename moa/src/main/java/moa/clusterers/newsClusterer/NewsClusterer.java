package moa.clusterers.newsClusterer;

import weka.core.EuclideanDistance;

//import com.yahoo.labs.samoa.instances.Instance;


import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

/***
 * Two level clusterer for a stream of strings. Creates 5 clusters at the top level
 * and 2 clusters within each of the top level topics. Uses euclidean distance to check for
 * novelty and reclusters at appropriate level when the related novelty container is full.
 * @author Gulnar
 *@author Guangru
 */
public class NewsClusterer {
    static NewsClusterer nc;
    //Current instance to be processed. Instances instead of instance to allow easy use of filter
    static Instances currentInstance;
    //All instances loaded from arff file
    static Instances inputInstances;
    //Instances used to create the top level featurespace
    static Instances featureInstances;
    public static final int NUM_GLOBAL_CLUSTERS = 5;
    //Instances used to create subtopic feature space
    static Instances[] subTopicFeatureSpace = new Instances[NUM_GLOBAL_CLUSTERS];
    //Clusterer to use for top level clustering
    static SimpleKMeans clustererTopLevel;
    //Clusterers to use for second level clustering
    static SimpleKMeans[] subClusterers = new SimpleKMeans[NUM_GLOBAL_CLUSTERS];
    //Filter to use for top level
    static FilterUtil filterTop;
    //Filters to use for sub topics
    static FilterUtil[] subTopicFilters = new FilterUtil[NUM_GLOBAL_CLUSTERS];
    //Distance funtion to use for top level
    static EuclideanDistance distFunction = new EuclideanDistance();
    //Distance function to use for lower level
    static EuclideanDistance[] subDistFunction = new EuclideanDistance[NUM_GLOBAL_CLUSTERS];
    //To store instances in each cluster
    static Instances[] bins = new Instances[NUM_GLOBAL_CLUSTERS];
    //To store novel items at top level
    static Instances novelty;
    //To store novel items at sub topic level
    static Instances[] subTopicNovelties = new Instances[NUM_GLOBAL_CLUSTERS];
    //to store the cluster radius at top level
    static double[] maxDistance;
    //to store cluster radius at sub topic level
    static double[][] subClusterDistances = new double[NUM_GLOBAL_CLUSTERS][2]; // Number of local clusters
    //to store centroids retrived at top level
    static Instances centroids;
    //To store centroids of sub topic clusterers
    static Instances[] subCentroids = new Instances[NUM_GLOBAL_CLUSTERS];
    //stats regarding reclustering
    static int numTopRecluster = 0;
    static int numSubRecluster = 0;
    //to store weather subclustering has been initialised for bins
    static boolean[] isSubClusterInit = new boolean[NUM_GLOBAL_CLUSTERS];


    static int topLevelFeatureSpaceSize = 50;
    static int subClusteringBinSize = 20;
    static int topNoveltySize = 100;
    static int subNoveltySize = 20;

    //threshold for top level novelty detection(proportion of cluster radius)
    static double topNoveltyThreshold = 0.8; // 0.5 to 1
    //threshold for sub topic level novelty detection(proportion of cluster radius)
    static double subNoveltyThreshold = 0.8; // 0.5 to 1

    static boolean IS_SAVING_FEATURE_SPACE = false;
    static String SAVE_FEATURE_PATH = "saveFeature_bbc.arff";

    public static void run(String fileName) {
        nc = new NewsClusterer();
        filterTop = new FilterUtil();
        //load arff file containing strings
        inputInstances = filterTop.loadARFF(fileName);
        //initialise instance bins for each topic
        for (int i = 0; i < bins.length; i++) {
            bins[i] = new Instances(inputInstances, 0, 0);

        }
        currentInstance = new Instances(inputInstances, 1);
        //get instances from feature space (could have collected from loop simulating stream)
        featureInstances = new Instances(inputInstances, 0, topLevelFeatureSpaceSize);
        if (IS_SAVING_FEATURE_SPACE) {
            filterTop.saveARFF(SAVE_FEATURE_PATH, featureInstances);
        }
        //create top level clusterer
        nc.clusterTopLevel(featureInstances);
        novelty = new Instances(inputInstances, 0, 0);
        //loop to simulate stream
        for (int i = topLevelFeatureSpaceSize; i < inputInstances.size(); i++) {
            nc.trainOnInstanceImpl(inputInstances.get(i));
        }
        System.out.println("" + topLevelFeatureSpaceSize + " , " + numTopRecluster + " , " + numSubRecluster);
    }

    /**
     * creates top level cluster using the given instances
     *
     * @param instances
     */
    public void clusterTopLevel(Instances instances) {
        filterTop = new FilterUtil();
        filterTop.initFilter(instances);
        featureInstances = filterTop.filterInstances(instances);
        clustererTopLevel = new SimpleKMeans();
        distFunction = new EuclideanDistance();

        // simple k-means
        try {
            clustererTopLevel.setPreserveInstancesOrder(true);
            distFunction.setInstances(featureInstances);
            clustererTopLevel.setNumClusters(NUM_GLOBAL_CLUSTERS);

            clustererTopLevel.buildClusterer(featureInstances);
            centroids = clustererTopLevel.getClusterCentroids();
            maxDistance = new double[5];
            int[] assignments = clustererTopLevel.getAssignments();
            int i = 0;
            for (int d : assignments) {
                bins[d].add(instances.get(i));
                double distance = distFunction.distance(featureInstances.get(i), centroids.get(d));
                if (maxDistance[d] < distance) {
                    maxDistance[d] = distance;
                }
                i++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Method to be called at the arrival of each instance after initial top level cluster is created
     *
     * @param inst
     */
    public void trainOnInstanceImpl(Instance inst) {

        currentInstance.add(inst);
        try { // filter and classify the coming inst into top level or sub-level
            Instances filtered = filterTop.filterInstances(currentInstance);
            int assignment = clustererTopLevel.clusterInstance(filtered.firstInstance());
            double d = distFunction.distance(filtered.firstInstance(), centroids.get(assignment));
            bins[assignment].add(inst);
            if ((maxDistance[assignment] * topNoveltyThreshold) < d) {
                novelty.add(inst);
                if (novelty.size() >= topNoveltySize) {
                    numTopRecluster++;
                    nc.clusterTopLevel(novelty);
                    novelty = new Instances(inputInstances, 0, 0);
                    for (int i = 0; i < bins.length; i++) {
                        subClusterers[i] = null;
                        bins[i] = new Instances(inputInstances, 0, 0);
                        isSubClusterInit[i] = false;
                    }
                }
            }

            //initialise sub clusterers if bin has enough examples
            if ((!isSubClusterInit[assignment]) && bins[assignment].size() > subClusteringBinSize) {
                subTopicFilters[assignment] = new FilterUtil();
                subTopicFilters[assignment].initFilter(bins[assignment]);
                subTopicFeatureSpace[assignment] = Filter.useFilter(bins[assignment], subTopicFilters[assignment].filter);
                subClusterers[assignment] = new SimpleKMeans();
                subClusterers[assignment].setPreserveInstancesOrder(true);
                subClusterers[assignment].buildClusterer(subTopicFeatureSpace[assignment]);
                subCentroids[assignment] = subClusterers[assignment].getClusterCentroids();
                int[] subAssignments = subClusterers[assignment].getAssignments();
                int i = 0;

                subClusterDistances[assignment] = new double[2];
                subDistFunction[assignment] = new EuclideanDistance();
                subDistFunction[assignment].setInstances(subTopicFeatureSpace[assignment]);
                for (int s : subAssignments) {
                    double distance = subDistFunction[assignment].distance(subTopicFeatureSpace[assignment].get(i), subCentroids[assignment].get(s));
                    if (subClusterDistances[assignment][s] < distance) {
                        subClusterDistances[assignment][s] = distance;
                    }
                    i++;
                }
                subTopicNovelties[assignment] = new Instances(inputInstances, 0, 0);
                isSubClusterInit[assignment] = true;
                //System.out.println("SubClusterInitDone");


            }
            //perform sub topic clustering if bin has been initilised
            else if (isSubClusterInit[assignment]) {
                //System.out.println("Bin size = " + bins[assignment].size());
                Instances forClusterer = subTopicFilters[assignment].filterInstances(currentInstance);
                int subClusterAssignment = subClusterers[assignment].clusterInstance(forClusterer.firstInstance());

                double sd = subDistFunction[assignment].distance(forClusterer.firstInstance(), subCentroids[assignment].get(subClusterAssignment));
                if ((subClusterDistances[assignment][subClusterAssignment] * subNoveltyThreshold) < sd) {
                    subTopicNovelties[assignment].add(inst);
                    if (subTopicNovelties[assignment].size() > 20) { // TODO random choice
                        numSubRecluster++;
                        subClusterers[assignment] = null;
                        bins[assignment] = new Instances(subTopicNovelties[assignment]);
                        subTopicNovelties[assignment] = new Instances(inputInstances, 0, 0);
                        isSubClusterInit[assignment] = false;
                    }
                }
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        currentInstance.remove(0);
    }
}