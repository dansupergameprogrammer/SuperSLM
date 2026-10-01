// Encode a committed UTF-8 reference prompt with an artifact's own tokenizer.
#include "superslm/artifact.h"
#include "superslm/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc % 2 != 0)
            throw std::runtime_error("usage: t3035_tokenize MODEL TEXT OUT_IDS [TEXT OUT_IDS ...]");
        superslm::SslmArtifact artifact;
        superslm::SslmError error{};
        if (superslm::SslmArtifact::OpenFromFile(argv[1], artifact, &error) != superslm::SslmStatus::Ok)
            throw std::runtime_error("artifact open failed");
        superslm::TokenizerView tokenizer;
        std::string why;
        if (!superslm::TokenizerView::Open(artifact, tokenizer, &why))
            throw std::runtime_error("tokenizer open failed: " + why);
        for (int pair = 2; pair < argc; pair += 2) {
            std::ifstream input(argv[pair], std::ios::binary);
            if (!input) throw std::runtime_error("prompt open failed");
            const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            const auto ids = tokenizer.Encode(text);
            if (ids.empty()) throw std::runtime_error("tokenizer returned no ids");
            std::ofstream output(argv[pair + 1], std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("ids open failed");
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i) output << ' ';
                output << ids[i];
            }
            output << '\n';
            output.flush();
            if (!output) throw std::runtime_error("ids write failed");
            std::cerr << argv[pair] << " token_ids=" << ids.size() << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "T-3035 tokenize: " << e.what() << '\n';
        return 1;
    }
}
