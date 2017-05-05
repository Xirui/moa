package moa.clusterers.newsClusterer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import java_cup.reduce_action;
import weka.core.EuclideanDistance;

//import com.yahoo.labs.samoa.instances.Instance;


import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.KMeans;
import moa.core.Measurement;
import moa.streams.ArffFileStream;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import moa.clusterers.streamkm.StreamKM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
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
	//Instances used to create subtopic feature space
	static Instances[] subTopicFeatureSpace = new Instances[5];
	//Clusterer to use for top level clustering
	static SimpleKMeans clustererTopLevel;
	//Clusterers to use for second level clustering
	static SimpleKMeans[] subClusterers = new SimpleKMeans[5];
	//Filter to use for top level 
	static FilterUtil filterTop;
	//Filters to use for sub topics
	static FilterUtil[] subTopicFilters = new FilterUtil[5];
	//Distance funtion to use for top level
	static EuclideanDistance distFunction = new EuclideanDistance();
	//Distance function to use for lower level
	static EuclideanDistance[] subDistFunction = new EuclideanDistance[5];
	//To store instances in each cluster
	static Instances[] bins = new Instances[5];
	//To store novel items at top level
	static Instances novelty;
	//To store novel items at sub topic level
	static Instances[] subTopicNovelties = new Instances[5];
	//to store the cluster radius at top level
	static double[] maxDistance;
	//to store cluster radius at sub topic level
	static double[][] subClusterDistances = new double[5][2];
	//to store centroids retrived at top level
	static Instances centroids;
	//To store centroids of sub topic clusterers
	static Instances[] subCentroids = new Instances[5]; 
	//stats regarding reclustering
	static int numTopRecluster = 0;
	static int numSubRecluster = 0;
	//to store weather subclustering has been initialised for bins
	static boolean[] isSubClusterInit = new boolean[5];
	//size of top level feature space -default 50
	static int topLevelFeatureSpaceSize = 50;
	//size of bins before sub topic clustering
	static int binSizeForSubClustering = 20;
	//size of top level novelty container
	static int topNoveltySize = 100;
	//size of sub topic novelty containers
	static int subNoveltySize = 20;
	//path to save feature space
	static String saveFeaturePath;
	//threshold for top level novelty detection(proportion of cluster radius)
	static double topNoveltyThreshold = 0.8;
	//threshold for sub topic level novelty detection(proportion of cluster radius)
	static double subNoveltyThreshold = 0.8;
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
	/**
	 * Main method of newsClusterer. Simulates a stream from a given arff file containing strings 
	 * and uses Kmeans clustering to form a 2 level cluster hierarchy
	 * @param args
	 */
	public static void main(String[] args) {
		//whether to save initial feature space
		boolean saveFeatureSpace = false;
		//get arguments 
		if (args.length < 1)
			System.out.println(USAGE);
		else {
			try{
				for (int c = 0; c<args.length; c++) {
					if(args[c].equals("-f")){
						topLevelFeatureSpaceSize = Integer.parseInt(args[c+1]);	
					}
					if(args[c].equals("-b")){
						binSizeForSubClustering = Integer.parseInt(args[c+1]);	
					}
					if(args[c].equals("-t")){
						topNoveltySize = Integer.parseInt(args[c+1]);	
					}
					if(args[c].equals("-s")){
						subNoveltySize = Integer.parseInt(args[c+1]);	
					}
					if(args[c].equals("-q")){
						saveFeaturePath = args[c+1];
						saveFeatureSpace = true;
					}
					if(args[c].equals("-n")){
						topNoveltyThreshold = Double.parseDouble(args[c+1]);
					}
					if(args[c].equals("-c")){
						subNoveltyThreshold = Double.parseDouble(args[c+1]);
					}
				}
			}
			catch(Exception e){
				System.out.println(USAGE);
			}
			nc = new NewsClusterer();
			filterTop = new FilterUtil();
			//load arff file containing strings
			inputInstances = filterTop.loadARFF(args[0]);	
			//initialise instance bins for each topic
			for(int i = 0; i<bins.length; i++){
				bins[i] = new Instances(inputInstances, 0, 0);

			}
			currentInstance = new Instances(inputInstances, 1);
			//get instances from feature space (could have collected from loop simulating stream)
			featureInstances = new Instances(inputInstances, 0, topLevelFeatureSpaceSize);
			if(saveFeatureSpace){
				filterTop.saveARFF(saveFeaturePath, featureInstances);
			}
			//create top level clusterer 
			nc.clusterTopLevel(featureInstances);
			novelty = new Instances(inputInstances, 0, 0);
			//loop to simulate stream
			for(int i = topLevelFeatureSpaceSize; i < inputInstances.size(); i++){
				nc.trainOnInstanceImpl(inputInstances.get(i));
			}
			//print final stats
			System.out.println("End novelty bin size = " + novelty.size());
			System.out.println("Number of global reclustring = " + numTopRecluster);
			System.out.println("Number of sub topic reclustring = " + numSubRecluster);
		}

	}
	/**
	 * creates top level cluster using the given instances
	 * @param instances
	 */
	public void clusterTopLevel(Instances instances){
		if(clustererTopLevel != null){
			double[] clusterSizes = clustererTopLevel.getClusterSizes();
			System.out.println("Cluster sizes when reclustering");
			for (int x = 0; x < clusterSizes.length; x++) {				
				System.out.println("Cluster " + x + " size = " + clusterSizes[x]);
			}
		}
		filterTop = new FilterUtil();
		filterTop.initFilter(instances);
		featureInstances = filterTop.filterInstances(instances);
		clustererTopLevel = new SimpleKMeans();
		distFunction = new EuclideanDistance();

		try {
			clustererTopLevel.setPreserveInstancesOrder(true);
			distFunction.setInstances(featureInstances);
			clustererTopLevel.setNumClusters(5);

			clustererTopLevel.buildClusterer(featureInstances);
			centroids = clustererTopLevel.getClusterCentroids();
			maxDistance = new double[5];
			int[] assignments = clustererTopLevel.getAssignments();
			int i = 0;
			for (int d : assignments) {
				bins[d].add(instances.get(i));
				double distance = distFunction.distance(featureInstances.get(i), centroids.get(d));
				if(maxDistance[d] < distance){
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
	 * @param inst
	 */
	public void trainOnInstanceImpl(Instance inst) {

		currentInstance.add(inst);
		try {
			Instances filtered = filterTop.filterInstances(currentInstance);
			double dist = 0.0;
			int assignment = clustererTopLevel.clusterInstance(filtered.firstInstance());
			double d =  distFunction.distance(filtered.firstInstance(), centroids.get(assignment));
			//System.out.println("Assignment = " + assignment);
			bins[assignment].add(inst);
			if((maxDistance[assignment]*topNoveltyThreshold)<d){
				novelty.add(inst);
				if(novelty.size() >= topNoveltySize){
					numTopRecluster++;
					nc.clusterTopLevel(novelty);
					novelty = new Instances(inputInstances,0,0);
					for(int i = 0; i<bins.length; i++){
						subClusterers[i] = null;
						bins[i] = new Instances(inputInstances, 0, 0);
						isSubClusterInit[i] = false;
					}
				}
			}
			//System.out.println("BinSize = " +bins[assignment].size());
			//initialise sub clusterers if bin has enough examples
			if((!isSubClusterInit[assignment]) && bins[assignment].size() > binSizeForSubClustering){
				subTopicFilters[assignment] = new FilterUtil();
				subTopicFilters[assignment].initFilter(bins[assignment]);
				subTopicFeatureSpace[assignment] = Filter.useFilter(bins[assignment], subTopicFilters[assignment].filter);
				subClusterers[assignment] = new SimpleKMeans();
				subClusterers[assignment].setPreserveInstancesOrder(true);
				subClusterers[assignment].setNumClusters(2);
				subClusterers[assignment].buildClusterer(subTopicFeatureSpace[assignment]);
				subCentroids[assignment] = subClusterers[assignment].getClusterCentroids();
				int[] subAssignments = subClusterers[assignment].getAssignments();
				int i = 0;

				subClusterDistances[assignment] = new double[2];
				subDistFunction[assignment] = new EuclideanDistance();
				subDistFunction[assignment].setInstances(subTopicFeatureSpace[assignment]);
				for (int s : subAssignments) {

					//System.out.println("SubAssignments = "+  s);
					//bins[d].add(featureInstances.get(i));
					double distance = subDistFunction[assignment].distance(subTopicFeatureSpace[assignment].get(i), subCentroids[assignment].get(s));
					if(subClusterDistances[assignment][s] < distance){
						subClusterDistances[assignment][s] = distance;
					}
					i++;
				}
				subTopicNovelties[assignment] = new Instances(inputInstances, 0, 0);
				isSubClusterInit[assignment] = true;
				//System.out.println("SubClusterInitDone");


			}
			//perform sub topic clustering if bin has been initilised
			else if(isSubClusterInit[assignment]){
				//System.out.println("Bin size = " + bins[assignment].size());
				Instances forClusterer = subTopicFilters[assignment].filterInstances(currentInstance);						
				int subClusterAssignment = subClusterers[assignment].clusterInstance(forClusterer.firstInstance());

				double sd =  subDistFunction[assignment].distance(forClusterer.firstInstance(), subCentroids[assignment].get(subClusterAssignment));
				if((subClusterDistances[assignment][subClusterAssignment] * subNoveltyThreshold) < sd){
					subTopicNovelties[assignment].add(inst);
					if(subTopicNovelties[assignment].size() > 15){
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