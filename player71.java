import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleEVD;
import org.ejml.data.DMatrixD1;

import java.util.Random;
import java.util.Properties;
import java.util.Arrays;

public class player71 implements ContestSubmission
{
	Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;

    // Main parameters
    private double sigma = 0.5; // Mutation standard deviation
    private int lambda = 100; // Number of samples per iteration
    private int mu = 25; // Number of samples selected
	
	public player71()
	{
		rnd_ = new Random();
	}
	
	public void setSeed(long seed)
	{
		// Set seed of algortihms random process
		rnd_.setSeed(seed);
	}

	public void setEvaluation(ContestEvaluation evaluation)
	{
		// Set evaluation problem used in the run
		evaluation_ = evaluation;
		
		// Get evaluation properties
		Properties props = evaluation.getProperties();
        // Get evaluation limit
        evaluations_limit_ = Integer.parseInt(props.getProperty("Evaluations"));
		// Property keys depend on specific evaluation
		// E.g. double param = Double.parseDouble(props.getProperty("property_name"));
        boolean isMultimodal = Boolean.parseBoolean(props.getProperty("Multimodal"));
        boolean hasStructure = Boolean.parseBoolean(props.getProperty("Regular"));
        boolean isSeparable = Boolean.parseBoolean(props.getProperty("Separable"));

		// Do sth with property values, e.g. specify relevant settings of your algorithm
        if(!hasStructure){
        	lambda = 2000;
        }
        if(!isMultimodal){
        	lambda = 50;
			mu = 6;
        }
    }
    
	public void run()
	{
		// Other parameters
		double weights[] = new double[mu];
		double sum = 0;
		double sum_squared = 0;
		for (int i = 0 ; i < mu; i++){
			double weight = Math.log(mu+0.5) - Math.log(i + 1);
    		weights[i] = weight;
    		sum += weight;
    		sum_squared += Math.pow(weight, 2);
		}
		for (int i = 0 ; i < mu; i++)
			weights[i] /= sum;
		double mueff = Math.pow(sum, 2)/sum_squared;
		double cc = (4+mueff/10) / (14 + 2*mueff/10);
		double cs = (mueff+2) / (15+mueff);
		double c1 = 2 / (Math.pow(11.3, 2)+mueff);
		double cmu = Math.min(1-c1, 2 * (mueff-2+1/mueff) / (Math.pow(12, 2) + mueff));
		double damps = 1 + 2*Math.max(0, Math.sqrt((mueff-1)/(11))-1) + cs;
	
		double chiN = Math.pow(10, 0.5*(1-1/(40)+1/(2100)));

		// Starting matrices
		SimpleMatrix pc = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
		SimpleMatrix ps = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
		SimpleMatrix covariance = SimpleMatrix.diag(new double[]{1,1,1,1,1,1,1,1,1,1});
		SimpleMatrix invsqrt_covariance = SimpleMatrix.diag(new double[]{1,1,1,1,1,1,1,1,1,1});
		SimpleMatrix mean = SimpleMatrix.random_DDRM(10,1,-5,5, rnd_);

		// Initialize counters
		int evals = 0;
		double best_score = 0;

		// Iterate until evaluation limit is reached
        while(evals<evaluations_limit_ - lambda){
        	// Generate offspring
        	//System.out.println(sigma);
        	Individual offspring[] = new Individual[lambda];
        	for (int i = 0; i < lambda; i++){
        		double new_genotype[] = ((DMatrixD1)mean.plus(SimpleMatrix.randomNormal(covariance, rnd_).scale(sigma)).getMatrix()).getData();
        		for (int j = 0; j < 10; j++)
        			new_genotype[j] = Math.min(5, Math.max(-5, new_genotype[j]));
        		SimpleMatrix genotype = new SimpleMatrix(10, 1, true, new_genotype); 
        		double fitness = (double) evaluation_.evaluate(((DMatrixD1)genotype.getMatrix()).getData());
        		if (fitness > best_score){
        			best_score = fitness;
        			//System.out.println("wow");
        			//System.out.println(best_score);
        		}
        		Individual child = new Individual(genotype, fitness);
        		offspring[i] = child;
        		evals++;
        	}
        	Arrays.sort(offspring);

        	// Update mean
        	SimpleMatrix old_mean = mean;
        	mean = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
        	for (int i = 0; i < mu; i++)
        		mean = mean.plus(offspring[lambda - 1 - i].getGenotype().scale(weights[i]));

        	// Update evolution paths
        	ps = ps.scale(1-cs).plus(invsqrt_covariance.mult(mean.minus(old_mean).divide(sigma)).scale(Math.sqrt(cs*(2-cs)*mueff)));
        	int hsig = ps.normF()/Math.sqrt(1-Math.pow(1-cs, 2*evals/lambda))/chiN < 1.4 + 2/11 ? 1 : 0;
        	pc = pc.scale(1-cc).plus(mean.minus(old_mean).divide(sigma).scale(Math.sqrt(cc*(2-cc)*mueff)).scale(hsig));

        	// Calculate artmp
        	SimpleMatrix artmp = offspring[lambda - 1].getGenotype().minus(old_mean);
        	for (int i = 1; i < mu; i++){
        		artmp = artmp.concatColumns(offspring[lambda - 1 - i].getGenotype().minus(old_mean));
        	}
        	artmp = artmp.scale(1/sigma);

        	// Update covariance matrix
        	covariance = covariance.scale(1-c1-cmu).plus(pc.mult(pc.transpose()).plus(covariance.scale(cc*(2-cc)*hsig)).scale(c1)).plus(artmp.mult(SimpleMatrix.diag(weights)).mult(artmp.transpose()));

        	// Update sigma
        	sigma = sigma * Math.exp((cs/damps)*(ps.normF()/chiN - 1));

        	// Enforce symmetry
    		covariance = covariance.plus(covariance.transpose()).scale(0.5);

    		// Decompose covariance and calculate invsqrt
    		SimpleEVD covariance_evd = covariance.eig();
    		double eigenvalues[] = new double[10];
    		for (int i = 0; i < 10; i++)
    			eigenvalues[i] = covariance_evd.getEigenvalue(i).getReal();
    		SimpleMatrix b = (SimpleMatrix)covariance_evd.getEigenVector(0);
    		for (int i = 1; i < 10; i++) {
				b = b.concatColumns((SimpleMatrix)covariance_evd.getEigenVector(i));		
			}
			invsqrt_covariance = b.mult(SimpleMatrix.diag(eigenvalues).invert()).mult(b.transpose());
        }
	}
}
