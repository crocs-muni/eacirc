#include <random>

template <class T, class Initializer, class Mutator, class ObjectiveFn>
struct LocalSearch {
private:
    Initializer _initializer;
    Mutator _mutator;
    ObjectiveFn _objective_fn;

public:
    template <class StopFn> T operator()(StopFn stop_fn)
    {
        T solution;
        _initializer(solution);
        return operator()(solution, stop_fn);
    }

    template <class StopFn> T operator()(T solution, StopFn stop_fn)
    {
        auto fitness = _objective_fn(solution);

        while (!stop_fn(fitness)) {
            auto neighbour = solution;

            _mutator(neighbour);
            auto neighbour_fitness = _objective_fn(neighbour);

            if (neighbour_fitness >= fitness) {
                solution = std::move(neighbour);
                fitness = neighbour_fitness;
            }
        }
        return solution;
    }
};
