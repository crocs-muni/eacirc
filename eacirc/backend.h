#pragma once

struct backend {
    virtual ~backend() = default;
};

struct initializer {
    virtual ~initializer() = default;

    virtual void apply(backend &) = 0;
};

struct mutator {
    virtual ~mutator() = default;

    virtual void apply(backend &) = 0;
};

struct evaluator {
    virtual ~evaluator() = default;

    virtual double apply(const backend &) = 0;
};