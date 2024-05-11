import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

class DynamicAssetAllocationModel:

    def __init__(self, initial_population, investment_horizon, goals, probabilities, irrs, asset_classes, dataset):
        self.population = initial_population
        self.investment_horizon = investment_horizon
        self.goals = goals
        self.probabilities = probabilities
        self.irrs = irrs
        self.asset_classes = asset_classes
        self.dataset = dataset
        self.num_assets = len(asset_classes.columns)
        self.initial_wealth = initial_wealth
        self.additional_investment = additional_investment

    def simulate_bond_returns(self, beta0, beta1, beta2, beta3, tau1, tau2, maturity_term):
        """
        Bond asset simulation using Nelson-Siegel-Svensson (NSS) model.
        """
        t = np.arange(1, self.investment_horizon + 1)

        # Adjust maturity based on the specified term
        if maturity_term == '1-3Y':
            t = np.minimum(t, 3)  # Cap t at 3 years for 1-3Y bonds
        elif maturity_term == '5-7Y':
            t = np.minimum(t, 7)  # Cap t at 7 years for 5-7Y bonds
        elif maturity_term == '10+Y':
            t = np.minimum(t, self.investment_horizon)  # Use investment horizon for 10+Y bonds

        # Calculate y component-wise
        term1 = beta0
        term2 = beta1 * ((1 - np.exp(-t / tau1)) / (t / tau1))
        term3 = beta2 * (((1 - np.exp(-t / tau1)) / (t / tau1)) - np.exp(-t / tau1))
        term4 = beta3 * (((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2))

        # Combine terms to compute y
        bond_returns = term1 + term2 + term3 + term4

        # Return y directly as bond returns (assuming bond return = y)
        return bond_returns


    def simulate_stock_returns(self, mu, sigma, initial_price, investment_horizon):
        dt = 1  # time step (annual)
        num_steps = investment_horizon
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=num_steps)
        stock_prices = np.zeros(num_steps + 1)
        stock_prices[0] = initial_price
        for i in range(num_steps):
            next_price = stock_prices[i] * (1 + returns[i])
            stock_prices[i + 1] = next_price
        stock_returns = np.diff(np.log(stock_prices))
        return stock_returns

    def generate_asset_returns(self):
        asset_returns = np.zeros((self.investment_horizon, self.num_assets))

        bond_assets = {
            'FTSE Europe Government 1-3Y Bond Index - Total Return': '1-3Y',
            'FTSE Europe Government 5-7Y Bond Index - Total Return': '5-7Y',
            'FTSE Europe Government 10+Y Bond Index - Total Return': '10+Y',
            'Bloomberg Global Agg Corporate Total Return Index Value Unhedged USD': None,
            'Bloomberg Global High Yield Total Return Index Value Unhedge': None
        }

        for i, asset_name in enumerate(self.asset_classes.columns):
            if asset_name in bond_assets:
                # Bond asset simulation using Nelson-Siegel-Svensson (NSS) model
                maturity_term = bond_assets[asset_name]
                bond_returns = self.simulate_bond_returns(
                    beta0=0.800291,
                    beta1=3.085896,
                    beta2=-2.977309,
                    beta3=6.852752,
                    tau1=3.056478,
                    tau2=10.834817,
                    maturity_term=maturity_term
                )
                asset_returns[:, i] = bond_returns
            else:
                # Stock asset simulation using Geometric Brownian Motion (GBM) model
                asset_returns[:, i] = self.simulate_stock_returns(
                    mu=0.05639,
                    sigma=0.03884225,
                    initial_price=100,  # Example initial price, adjust as needed
                    investment_horizon=self.investment_horizon
                )

        return asset_returns

    def simulate_bond_returns(self, beta0, beta1, beta2, beta3, tau1, tau2, maturity_term):
        t = np.arange(1, self.investment_horizon + 1)

        if maturity_term == '1-3Y':
            t = np.minimum(t, 3)  # Cap t at 3 years for 1-3Y bonds
        elif maturity_term == '5-7Y':
            t = np.minimum(t, 7)  # Cap t at 7 years for 5-7Y bonds
        elif maturity_term == '10+Y':
            t = np.minimum(t, self.investment_horizon)  # Use investment horizon for 10+Y bonds

        y = beta0 + beta1 * ((1 - np.exp(-t / tau1)) / (t / tau1)) + beta2 * (((1 - np.exp(-t / tau1)) / (t / tau1)) - np.exp(-t / tau1)) + beta3 * (((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2))
        bond_returns = np.diff(y)
        bond_returns = np.concatenate(([0], bond_returns))
        return bond_returns


    def calculate_portfolio_performance(self, weights):
        # Calculate portfolio performance over time
        portfolio_values = [self.initial_wealth]
        asset_returns = self.generate_asset_returns()  # Use self.asset_classes and self.dataset

        for year in range(1, self.investment_horizon + 1):
            # Calculate returns for each asset based on weights
            year_returns = asset_returns[year-1]
            portfolio_return = np.dot(weights, year_returns)

            # Calculate portfolio value for this year
            if year == 1:
                # First year: initial wealth + additional investment
                portfolio_value = portfolio_values[0]
            else:
                # Subsequent years: previous year's portfolio value + investment + returns
                portfolio_value = portfolio_values[-1] * (1 + portfolio_return) + self.additional_investment

            # Check and withdraw for each goal
            for goal in self.goals:
                goal_year = goal['year']
                if year == goal_year:
                    withdrawal_amount = goal['amount']
                    portfolio_value -= withdrawal_amount

            # Append the portfolio value for the current year
            portfolio_values.append(portfolio_value)

        return np.array(portfolio_values)

    def fitness_function(self, weights):
        # Evaluate fitness based on the primary goal of minimum success probability and secondary goal of IRR shortfall

        # Calculate portfolio performance over time
        final_portfolio_values = self.calculate_portfolio_performance(weights)

        # Initialize variables to track success and IRR shortfall
        success_count = 0
        total_irr_shortfall = 0

        # Evaluate each goal
        for goal in self.goals:
            goal_year = goal['year']
            goal_amount = goal['amount']

            if goal_year <= self.investment_horizon:
                portfolio_value_at_goal_year = final_portfolio_values[goal_year]

                # Check if the goal is met
                if portfolio_value_at_goal_year >= goal_amount:
                    success_count += 1

                # Calculate IRR shortfall if the goal is not met
                else:
                    irr_shortfall = goal_amount - portfolio_value_at_goal_year
                    total_irr_shortfall += irr_shortfall

        # Calculate success probability (fraction of goals met)
        success_probability = success_count / len(self.goals)

        # Fitness score prioritizes success probability, then minimizes IRR shortfall if successful
        fitness_score = success_probability  # Primary goal: maximize success probability

        if success_probability > 0:
            # If success probability is achieved, minimize expected shortfall of IRR
            fitness_score -= total_irr_shortfall / success_probability

        return fitness_score

    def evolve_population(self, num_generations, population_size, initial_mutation_rate, final_mutation_rate):
        # Genetic algorithm to evolve the population
        num_assets = len(self.asset_classes.columns)
        population = deepcopy(self.population)
        optimal_weights_history = []

        for generation in range(num_generations):
            # Evaluate fitness of each portfolio
            fitness_scores = [self.fitness_function(weights) for weights in population]

            # Select top portfolios based on fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = int(population_size * 0.3)  # Number of elite to keep
            elite_indices = sorted_indices[:min(elite_count, len(population))]  # Keep top elite_count as elite

            elite_indices = np.array(elite_indices).flatten()

            # Create new population through selection, crossover, and mutation
            new_population = []

            while len(new_population) < population_size:
                if len(elite_indices) < 2:
                    raise ValueError("Elite indices size is less than 2. Cannot perform crossover.")

                # Select two parents
                parent_indices = np.random.choice(elite_indices, size=2, replace=False)

                parent1_weights = population[parent_indices[0] % len(population)]  # Wrap index with modulo to stay within bounds
                parent2_weights = population[parent_indices[1] % len(population)]  # Wrap index with modulo to stay within bounds

                child_weights = self.crossover(parent1_weights, parent2_weights)
                current_mutation_rate = self.interpolate_mutation_rate(initial_mutation_rate, final_mutation_rate, generation, num_generations)
                child_weights = self.mutate(child_weights, current_mutation_rate)

                new_population.append(child_weights)

            population = new_population

            optimal_weights = population[np.argmax(fitness_scores)]
            optimal_weights_history.append(optimal_weights)

        self.population = population
        self.optimal_weights_history = optimal_weights_history

    def interpolate_mutation_rate(self, initial_rate, final_rate, current_generation, total_generations):
        progress = current_generation / total_generations
        return initial_rate + (final_rate - initial_rate) * progress

    def crossover(self, parent1, parent2):
        # Uniform crossover
        child = []
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def mutate(self, portfolio_weights, mutation_rate):
        # Randomly mutate some weights
        mutated_weights = []
        for weight in portfolio_weights:
            if np.random.rand() < mutation_rate:
                mutated_weights.append(np.random.rand())
            else:
                mutated_weights.append(weight)
        return mutated_weights

    def print_success_probability(self):
        if not hasattr(self, 'optimal_weights_history'):
            raise ValueError("Optimal portfolio weights history is not available. Please run the genetic algorithm.")

        success_count_by_goal = {goal['year']: 0 for goal in self.goals}

        for weights in self.optimal_weights_history:
            for goal in self.goals:
                goal_year = goal['year']
                goal_amount = goal['amount']
                if goal_year <= self.investment_horizon:
                    final_portfolio_values = self.calculate_portfolio_performance(weights)
                    portfolio_value_at_goal_year = final_portfolio_values[goal_year]
                    if portfolio_value_at_goal_year >= goal_amount:
                        success_count_by_goal[goal_year] += 1

        success_probability_by_goal = {
            goal['year']: success_count_by_goal[goal['year']] / len(self.optimal_weights_history) for goal in self.goals
        }

        for goal in self.goals:
            goal_year = goal['year']
            print(f"Success Probability for Goal at Year {goal_year}: {success_probability_by_goal[goal_year]:.2%}")


    def meets_investment_goals(self, weights):
        final_portfolio_values = self.calculate_portfolio_performance(weights)

        for goal in self.goals:
            goal_year = goal['year']
            goal_amount = goal['amount']
            if goal_year <= self.investment_horizon:
                portfolio_value_at_goal_year = final_portfolio_values[goal_year]
                if portfolio_value_at_goal_year < goal_amount:
                    return False

        return True

    def select_optimal_portfolio(self):
        # Select the optimal dynamic asset allocation
        if not self.population:
            raise ValueError("Population is empty. Please run the genetic algorithm to populate the population.")

        # Evaluate fitness of each portfolio in the population
        fitness_scores = [self.fitness_function(weights) for weights in self.population]

        if len(fitness_scores) == 0 or np.any(np.isnan(fitness_scores)):
            raise ValueError("Fitness scores are invalid or empty. Cannot determine optimal portfolio.")

        try:
            best_portfolio_index = np.nanargmax(fitness_scores)
        except ValueError:
            raise ValueError("Unable to determine the best portfolio index. Check fitness scores and population.")

        if best_portfolio_index < 0 or best_portfolio_index >= len(self.population):
            raise ValueError(f"Invalid best_portfolio_index: {best_portfolio_index}. "
                             f"It should be within the range [0, {len(self.population) - 1}].")

        optimal_weights = self.population[best_portfolio_index]
        asset_names = list(self.asset_classes.columns)

        optimal_weights /= np.sum(optimal_weights)

        return optimal_weights

    def plot_portfolio_paths(self, num_samples=10):
        plt.figure(figsize=(12, 8))

        for i in range(num_samples):
            weights = np.random.rand(len(self.asset_classes.columns))
            weights /= np.sum(weights)  # Normalize to ensure sum(weights) = 1
            portfolio_values = self.calculate_portfolio_performance(weights)
            plt.plot(range(self.investment_horizon + 1), portfolio_values, alpha=0.5, label=f'Portfolio {i+1}')

        optimal_weights = self.select_optimal_portfolio()
        optimal_portfolio_values = self.calculate_portfolio_performance(optimal_weights)
        plt.plot(range(self.investment_horizon + 1), optimal_portfolio_values, 'r-', linewidth=2, label='Optimal Portfolio')

        plt.xlabel('Years')
        plt.ylabel('Portfolio Value')
        plt.title('Portfolio Paths')
        custom_legend = [
            Line2D([0], [0], color='red', linewidth=2),  # Red line for optimal portfolio
            Line2D([0], [0], color='blue', linewidth=1),  # Blue line for sampled portfolios (example)
        ]
        legend_labels = ['Optimal Portfolio'] + ['Other Portfolios']

        # Show legend with custom legend entries and labels
        plt.legend(custom_legend, legend_labels)

        plt.show()

    def plot_optimal_weights_over_time(self):
        if not hasattr(self, 'optimal_weights_history'):
            raise ValueError("Optimal portfolio weights history is not available. Please run the genetic algorithm.")

        optimal_weights_history = np.array(self.optimal_weights_history)
        num_assets = len(self.asset_classes.columns)
        asset_names = list(self.asset_classes.columns)

        plt.figure(figsize=(12, 8))
        for asset_index in range(num_assets):
            plt.plot(range(len(optimal_weights_history)), optimal_weights_history[:, asset_index], label=asset_names[asset_index])

        plt.xlabel('Generation')
        plt.ylabel('Portfolio Weight')
        plt.title('Optimal Portfolio Weights Over Generations')
        plt.legend()
        plt.show()

    def plot_annual_optimal_weights(self):
        if not hasattr(self, 'optimal_weights_history'):
            raise ValueError("Optimal portfolio weights history is not available. Please run the genetic algorithm.")

        optimal_weights_history = np.array(self.optimal_weights_history)
        num_assets = len(self.asset_classes.columns)
        asset_names = list(self.asset_classes.columns)
        investment_horizon = self.investment_horizon

        # Initialize a matrix to store annual optimal weights
        annual_optimal_weights = np.zeros((investment_horizon, num_assets))

        # Aggregate optimal weights by year
        for year in range(investment_horizon):
            year_weights = optimal_weights_history[year::investment_horizon, :]  # Select weights for each year
            annual_optimal_weights[year, :] = np.mean(year_weights, axis=0)  # Calculate mean weights for the year

        annual_optimal_weights = annual_optimal_weights / np.sum(annual_optimal_weights, axis=1, keepdims=True)

        # Plot annual optimal portfolio weights
        plt.figure(figsize=(12, 8))
        for asset_index in range(num_assets):
            plt.plot(range(1, investment_horizon + 1), annual_optimal_weights[:, asset_index], label=asset_names[asset_index])

        plt.xlabel('Investment Year')
        plt.ylabel('Portfolio Weight')
        plt.title('Annual Optimal Portfolio Weights Over Investment Horizon')
        plt.legend()
        plt.show()




initial_wealth = 150000

additional_investment = 5000

investment_horizon = 30


goals = [
    {'year': 10, 'amount': 10000},
    {'year': 11, 'amount': 10000},
    {'year': 12, 'amount': 10000},
    {'year': 13, 'amount': 10000},
    {'year': 20, 'amount': 20000},
    {'year': 21, 'amount': 20000},
    {'year': 22, 'amount': 20000},
    {'year': 28, 'amount': 50000},
]




# Define the initial population of portfolios
num_portfolios = 100
num_assets = len(asset_classes.columns)
initial_population = np.random.rand(num_portfolios, num_assets)

# Define probabilities and IRRs
probabilities = {'min_success': 0.975, 'joint_achievement': None}
irrs = {'average_irr_worst_cases': None}

model = DynamicAssetAllocationModel(initial_population, investment_horizon, goals, probabilities, irrs, asset_classes, dataset)
model.evolve_population(num_generations=150, population_size=200, initial_mutation_rate=0.4, final_mutation_rate=0.4)

model.plot_annual_optimal_weights()
model.plot_portfolio_paths(num_samples=1000)
model.print_success_probability()
