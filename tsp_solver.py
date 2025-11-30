import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Reading in the data
def read_data():
    store_data = pd.read_csv('data/storedata.csv')
    travel_time = pd.read_csv('data/storeTraveltime.csv', index_col=0)
    travel_cost = pd.read_csv('data/storeTravelcost.csv', index_col=0)
    return store_data, travel_time, travel_cost

# Storing the data for the problem
def create_data_model(travel_time):
    data = {}
    time_matrix = travel_time.values.astype(int).tolist() # Converted to integer for OR-tools
    data['time_matrix'] = time_matrix
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

# Solving the TSP using OR tools
def solve_tsp_ortools(travel_time):
    data = create_data_model(travel_time)
    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']), data['num_vehicles'], data['depot']
    )
    
    routing = pywrapcp.RoutingModel(manager) # The routing model
    
    # Creating and registering a transit callback
    def time_callback(from_index, to_index):
        # Returning the time between the two nodes
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Defining the cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30  
    
    # Solving the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Extracting the solution
    if solution:
        route = extract_solution(manager, routing, solution)
        return route
    else:
        print("No solution found!")
        return None

def extract_solution(manager, routing, solution):
    """Extracts route from OR-Tools solution"""
    index = routing.Start(0)
    route = [manager.IndexToNode(index)]
    
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
    
    return route

def calculate_route_stats(route, travel_time_matrix, travel_cost_matrix):
    """Calculate total time and cost for a route"""
    total_time = 0
    total_cost = 0
    
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        total_time += travel_time_matrix.iloc[from_node, to_node]
        total_cost += travel_cost_matrix.iloc[from_node, to_node]
    
    return total_time, total_cost

def main():
    store_data, travel_time, travel_cost = read_data()
    
    # Solving TSP using OR-Tools
    print("Solving TSP with OR-Tools...")
    route = solve_tsp_ortools(travel_time)
    
    if route:
        total_time, total_cost = calculate_route_stats(route, travel_time, travel_cost)
        print(f"Total driving time: {total_time} minutes")
        print(f"Total cost: €{total_cost:.2f}")
        print(f"Number of locations visited: {len(route)-1}")  # -1 because depot appears twice
        print(f"Route: {route}")
        
        # Saving the solution to file
        with open('1a.txt', 'w') as f:
            f.write(' '.join(map(str, route)))
        
        return route, total_time, total_cost
    else:
        print("Failed to find a solution")
        return None, None, None

if __name__ == "__main__":
    route, total_time, total_cost = main()