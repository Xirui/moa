package moa.clusterers.newsClusterer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
/**
 * Class to filter instances to use for NewsClusterer. Uses StringToWordVector filter
 * @author gulnar
 *@author Guangru
 */
public class FilterUtil {
	StringToWordVector filter;
	public FilterUtil(){
		filter = new StringToWordVector();
	}
	/**
	 * Loads an ARFF file into an instances object.
	 * @param fileName The name of the file to be loaded.
	 */
	public Instances loadARFF(String fileName) {

		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffReader arff = new ArffReader(reader);
			Instances instances = arff.getData();
//			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
			return instances;
		}
		catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
		return null;
	}
	/**
	 * Initialise the filter 
	 * @param instances
	 */
	public void initFilter(Instances instances) {
		// outputInstances = inputInstances;
		try {

			// Set the tokenizer -> split the article into individual word
			NGramTokenizer tokenizer = new NGramTokenizer();
			tokenizer.setNGramMinSize(1);
			tokenizer.setNGramMaxSize(1);
			tokenizer.setDelimiters("\\W");

			// Set the filter

			filter.setTokenizer(tokenizer);
			filter.setInputFormat(instances);
			filter.setWordsToKeep(500);
			filter.setStemmer(new weka.core.stemmers.LovinsStemmer());
			filter.setStopwordsHandler(new weka.core.stopwords.Rainbow());
			filter.setIDFTransform(true);
			filter.setTFTransform(true);
			filter.setDoNotOperateOnPerClassBasis(true);
			filter.setLowerCaseTokens(true);
		}
		catch (Exception e) {
			System.out.println("Problem found when training");
		}
	}
	/**
	 * 
	 * @param instances to filter
	 * @return Filtered instances
	 */
	public Instances filterInstances(Instances inputInstances){
		// Filter the input instances into the output ones
		try {
			Instances outputInstances = Filter.useFilter(inputInstances,filter);
			return outputInstances;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return inputInstances;
	}
	/**
	 * Save an instances object into an ARFF file.
	 * @param fileName The name of the file to be saved.
	 */	
	public void saveARFF(String fileName, Instances instances) {
	
		try {
			PrintWriter writer = new PrintWriter(new FileWriter(fileName));
			writer.print(instances);
			System.out.println("===== Saved dataset: " + fileName + " =====");
			writer.close();
		}
		catch (IOException e) {
			System.out.println("Problem found when writing: " + fileName);
		}
	}
}
