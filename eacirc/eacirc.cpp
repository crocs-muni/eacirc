#include "eacirc.h"
#include "circuit/module.h"
#include "finisher.h"
#include "projects/file.h"

Eacirc::Eacirc(const std::string)
    : _main_generator(std::random_device()())
    , _tv_size(16)
    , _num_of_tvs(500)
    , _num_of_epochs(300)
    , _change_frequency(100)
    , _significance_level(5) {
    _stream_A = std::make_unique<FileStream>("sha3_stream1.bin");
    _stream_B = std::make_unique<FileStream>("sha3_stream2.bin");

    _backend = circuit::Module::get_backend();
}

void Eacirc::run() {
    std::vector<double> pvalues;
    pvalues.reserve(_num_of_epochs);

    Dataset ins_A{_tv_size, _num_of_tvs};
    Dataset ins_B{_tv_size, _num_of_tvs};

    std::unique_ptr<Solver> solver = _backend->solver(_main_generator());

    for (size_t i = 0; i != _num_of_epochs; ++i) {
        _stream_A->read(ins_A);
        _stream_B->read(ins_B);

        solver->replace_datasets(ins_A, ins_B);

        pvalues.emplace_back(solver->reevaluate());
        solver->run(_change_frequency);

        std::cout << "epoch " << i << " finished" << std::endl;
    }

    Finisher::ks_test_finish(pvalues, _significance_level);
}
