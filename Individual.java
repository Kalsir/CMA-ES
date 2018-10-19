import org.ejml.simple.SimpleMatrix;

public class Individual implements Comparable<Individual>{
	private SimpleMatrix genotype;
	private double fitness;
	private SimpleMatrix random_normal_vec;

	public Individual(SimpleMatrix genotype, double fitness, SimpleMatrix random_normal_vec) {
		super();
		this.genotype = genotype;
		this.fitness = fitness;
		this.random_normal_vec =random_normal_vec;
	}

	public int compareTo(Individual individual) {
		if (this.fitness > individual.fitness)
			return 1;
		else if (this.fitness == individual.fitness)
			return 0;
		else
			return -1;
	}

	public SimpleMatrix getGenotype() {
		return this.genotype;
	}

	public  double getFitness() {
		return this.fitness;
	}

	public SimpleMatrix getRandomNormalVec() {
		return this.random_normal_vec;
	}
}