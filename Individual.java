import org.ejml.simple.SimpleMatrix;

public class Individual implements Comparable<Individual>{
	private SimpleMatrix genotype;
	private double fitness;

	public Individual(SimpleMatrix genotype, double fitness) {
		super();
		this.genotype = genotype;
		this.fitness = fitness;
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
}