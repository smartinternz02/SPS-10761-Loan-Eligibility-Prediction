package org.ml;

import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
public class LogRegression {
	
	public static Instances getInstances(String filename)
	{
		DataSource source;
		Instances dataset=null;
		try {
			source = new DataSource(filename);
			dataset = source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes()-1);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			
		}
		
		return dataset;
	}
	
	public static void main(String[] args) throws Exception
	{
		Instances data=getInstances("D:\\java ml\\Loan_predictor.csv");
		Instances test=getInstances("D:\\java ml\\test.csv");
		System.out.println(data.size());
		Classifier classifier = new weka.classifiers.functions.Logistic();

		classifier.buildClassifier(data);
		
		
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(classifier, data);
		System.out.println("** Logistic Regression Evaluation with Datasets **");
		System.out.println(eval.toSummaryString());
		
		
		
		double confusion[][]=eval.confusionMatrix();
		System.out.println("Confusion matrix:");
		for(double[] row: confusion)
			System.out.println(Arrays.toString(row));
		
		
		System.out.println("Area under the curve");
		System.out.println(eval.areaUnderROC(0));
		
		System.out.println(eval.getAllEvaluationMetricNames());
		
		System.out.println("Recall : ");
		System.out.println(Math.round(eval.recall(1)*100.0)/100.0);
		
		System.out.println("Precision: ");
		
		System.out.println(Math.round(eval.precision(1)*100.0)/100.0);
		System.out.print("F1 score:");
		System.out.println(Math.round(eval.fMeasure(1)*100.0)/100.0);
		
		System.out.print("Accuracy:");
		double acc = eval.correct()/(eval.correct()+ eval.incorrect());
		System.out.println(Math.round(acc*100.0)/100.0);
		
		
	}
}
