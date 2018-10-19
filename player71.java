import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleEVD;
import org.ejml.data.DMatrixD1;

import java.util.Random;
import java.util.Properties;
import java.util.Arrays;

// Based on The CMA Evolution Strategy: A Tutorial
public class player71 implements ContestSubmission
{
	Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;

    // Main parameters
    private double sigma = 0.5; // Initial mutation standard deviation
    private int lambda = 100; // Number of samples per iteration
    private int mu = 50; // Number of samples selected

    // Booleans for function properties
    private boolean isMultimodal = false;
    private boolean hasStructure = false;
    private boolean isSeparable = false;
	
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
		Properties prop_s = evaluation.getProperties();
        // Get evaluation limit
        evaluations_limit_ = Integer.parseInt(prop_s.getProperty("Evaluations"));
		// Property keys depend on specific evaluation
		// E.g. double param = Double.parseDouble(prop_s.getProperty("property_name"));
        isMultimodal = Boolean.parseBoolean(prop_s.getProperty("Multimodal"));
        hasStructure = Boolean.parseBoolean(prop_s.getProperty("Regular"));
        isSeparable = Boolean.parseBoolean(prop_s.getProperty("Separable"));

		// Do sth with property values, e.g. specify relevant settings of your algorithm
        if(!isMultimodal){
        	lambda = 8;
			mu = 4;
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
		double mu_eff = Math.pow(sum, 2)/sum_squared;
		double c_c = (4+mu_eff/10) / (14 + 2*mu_eff/10);
		double c_s = (mu_eff+2) / (15+mu_eff);
		double c_1 = 2 / (Math.pow(11.3, 2)+mu_eff);
		double c_mu = Math.min(1-c_1, 2 * (mu_eff-2+1/mu_eff) / (Math.pow(12, 2) + mu_eff));
		double damp_s = 1 + 2*Math.max(0, Math.sqrt((mu_eff-1)/(11))-1) + c_s;
		double chiN = Math.pow(10, 0.5)*(1-1/40+1/2100);

		// Starting matrices
		SimpleMatrix p_c = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
		SimpleMatrix p_s = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
		SimpleMatrix b = SimpleMatrix.identity(10);
		SimpleMatrix d = SimpleMatrix.identity(10);
		SimpleMatrix covariance = SimpleMatrix.identity(10);
		SimpleMatrix xmean = SimpleMatrix.random_DDRM(10,1,-5,5, rnd_);

		// Initialize counters
		int evals = 0;
		int current_evals = 0;
		double best_score = 0;

		double best_per_generation[] = new double[10];

		// Iterate until evaluation limit is reached
        while(evals<evaluations_limit_ - lambda){
        	// Generate offspring
        	Individual offspring[] = new Individual[lambda];
        	for (int i = 0; i < lambda; i++){
        		double random_normals[] = new double[10];
        		for (int j = 0; j < 10; j++)
        			random_normals[j] = rnd_.nextGaussian();
        		SimpleMatrix random_normal_vec = new SimpleMatrix(10,1,true, random_normals);
        		double new_genotype[] = ((DMatrixD1)xmean.plus(b.mult(d).mult(random_normal_vec).scale(sigma)).getMatrix()).getData();
        		for (int j = 0; j < 10; j++)
        			new_genotype[j] = Math.min(5, Math.max(-5, new_genotype[j]));
        		SimpleMatrix genotype = new SimpleMatrix(10, 1, true, new_genotype); 
        		double fitness = (double) evaluation_.evaluate(((DMatrixD1)genotype.getMatrix()).getData());
        		if (fitness > best_score){
        			best_score = fitness;
        			//System.out.println("wow");
        			//System.out.println(best_score);
        			//System.out.println(evals);
        		}
        		Individual child = new Individual(genotype, fitness, random_normal_vec);
        		offspring[i] = child;
        		evals++;
        		current_evals++;
        	}
        	Arrays.sort(offspring);

        	for (int i = 0; i < 9; i++)
                best_per_generation[i] = best_per_generation[i+1];
            best_per_generation[9] = offspring[lambda-1].getFitness();

        	// Update xmean
        	xmean = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
        	for (int i = 0; i < mu; i++)
        		xmean = xmean.plus(offspring[lambda - 1 - i].getGenotype().scale(weights[i]));
        	SimpleMatrix zmean = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
        	for (int i = 0; i < mu; i++)
        		zmean = zmean.plus(offspring[lambda - 1 - i].getRandomNormalVec().scale(weights[i])); 

        	// Update evolution paths
        	p_s = p_s.scale(1-c_s).plus(b.mult(zmean).scale(Math.sqrt(c_s*(2-c_s)*mu_eff)));
        	int hsig = p_s.normF()/Math.sqrt(1-Math.pow(1-c_s, 2*evals/lambda))/chiN < 1.4 + 2/11 ? 1 : 0;
        	p_c = p_c.scale(1-c_c).plus(b.mult(d).mult(zmean).scale(Math.sqrt(c_c*(2-c_c)*mu_eff)).scale(hsig));

			// Update covariance matrix        	
        	SimpleMatrix temp = offspring[lambda - 1].getRandomNormalVec();
        	for (int i = 1; i < mu; i++){
        		temp = temp.concatColumns(offspring[lambda - 1 - i].getRandomNormalVec());
        	}
        	temp = b.mult(d).mult(temp);
        	SimpleMatrix rank_one_update = p_c.mult(p_c.transpose()).plus(covariance.scale(c_c*(2-c_c)*(1-hsig))).scale(c_1);
          	SimpleMatrix rank_mu_update = temp.mult(SimpleMatrix.diag(weights)).mult(temp.transpose()).scale(c_mu);
        	covariance = covariance.scale(1-c_1-c_mu).plus(rank_one_update).plus(rank_mu_update);

        	// Update sigma (including weird hack that somehow makes Schaffers F7 work)
        	if (isMultimodal && hasStructure)
        		sigma = Math.exp((c_s/damp_s)*(p_s.normF()/chiN - 1));
        	else
        		sigma = sigma * Math.exp((c_s/damp_s)*(p_s.normF()/chiN - 1));

        	// Enforce symmetry
    		covariance = covariance.plus(covariance.transpose()).scale(0.5);

    		// Decompose covariance and calculate invsqrt
    		SimpleEVD covariance_evd = covariance.eig();
    		double eigenvalues[] = new double[10];
    		for (int i = 0; i < 10; i++)
    			eigenvalues[i] = Math.sqrt(covariance_evd.getEigenvalue(i).getReal());
    		b = (SimpleMatrix)covariance_evd.getEigenVector(0);
    		for (int i = 1; i < 10; i++) {
				b = b.concatColumns((SimpleMatrix)covariance_evd.getEigenVector(i));		
			}
			d = SimpleMatrix.diag(eigenvalues);

			
			// Restart if global optimum not found after 50000 evaluations for Katsuura
			// Restart if BentCigar not climbing fast enough
			if ((current_evals > 50000 && !hasStructure) || (best_score < 9.9999 && current_evals > 1500 && !isMultimodal)){
				p_c = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
				p_s = new SimpleMatrix(10,1,true, new double[]{0,0,0,0,0,0,0,0,0,0});
				b = SimpleMatrix.identity(10);
				d = SimpleMatrix.identity(10);
				covariance = SimpleMatrix.identity(10);
				xmean = SimpleMatrix.random_DDRM(10,1,-5,5, rnd_);
				best_score = 0;
				current_evals = 0;
			}
        }
	}
}