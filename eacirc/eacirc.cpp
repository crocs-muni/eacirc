#include "eacirc.h"
#include "circuit/backend.h"
#include "finisher.h"
#include "projects/file.h"

Eacirc::Eacirc(const std::string)
    : _num_of_tvs(128000), _num_of_epochs(0), _change_frequency(100), _significance_level(5) {
    _stream_A = std::make_unique<FileStream<16>>("stream_a.bin");
    _stream_B = std::make_unique<FileStream<16>>("stream_b.bin");
}

Eacirc::~Eacirc() = default;

void Eacirc::run() {
    std::vector<double> pvalues;
    pvalues.reserve(_num_of_epochs);

    Data<16, 1> data(_num_of_tvs);

    for (unsigned i = 0; i != _num_of_epochs; ++i) {
        _stream_A->read(data.ins_A);
        _stream_B->read(data.ins_B);

        _solver->data(&data);

        pvalues.emplace_back(_solver->reevaluate());
        _solver->run(_change_frequency);
    }

    Finisher::ks_test_finish(pvalues, _significance_level);
}
