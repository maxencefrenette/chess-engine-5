#include "probe/probe.h"
#include <cstdio>
#include <memory>
#include <stdexcept>

extern "C" {
void* ce5_tb_open(const char* directory, char* error, size_t capacity) noexcept {
    try {
        auto tables = std::make_unique<Probe_Tables>();
        const std::filesystem::path root(directory);
        if (!std::filesystem::is_directory(root / "wdl"))
            throw std::runtime_error("missing WDL directory");
        tables->add_wdl_path(root / "wdl");
        tables->add_wdl_path(root / "castling" / "wdl");
        if (!std::filesystem::is_directory(root / "dtz"))
            throw std::runtime_error("missing DTZ directory");
        tables->add_dtz_path(root / "dtz");
        tables->add_dtz_path(root / "castling" / "dtz");
        tables->rescan();
        return tables.release();
    } catch (const std::exception& e) {
        std::snprintf(error, capacity, "%s", e.what());
        return nullptr;
    }
}

void ce5_tb_close(void* tables) noexcept {
    delete static_cast<Probe_Tables*>(tables);
}

size_t ce5_tb_largest(void* tables) noexcept {
    return static_cast<Probe_Tables*>(tables)->largest();
}

// Returns the side-to-move WDL class adjusted with DTZ and the actual clock.
int ce5_tb_wdl(void* tables, const char* fen, unsigned ep, unsigned clock,
               char* error, size_t capacity) noexcept {
    try {
        const Position position = Position::from_fen(fen);
        if (position.piece_bb(WHITE_KING).num_set_bits() != 1 ||
            position.piece_bb(BLACK_KING).num_set_bits() != 1 ||
            ((position.piece_bb(WHITE_PAWN) | position.piece_bb(BLACK_PAWN)) & (RANK_1 | RANK_8)) ||
            !position.castling_rights_are_consistent() || !position.is_legal())
            throw std::runtime_error("invalid tablebase position");
        if (position.occupied().num_set_bits() == 2)
            return static_cast<int>(WDL_Entry::DRAW);
        const auto result = static_cast<Probe_Tables*>(tables)->probe(
            position, ep == 64 ? SQ_END : static_cast<Square>(ep), clock);
        if (result.status != Probe_Result::Status::OK || !result.has_wdl50 ||
            result.wdl50 == WDL_Entry::ILLEGAL)
            throw std::runtime_error("missing WDL/DTZ table or invalid probe");
        return static_cast<int>(result.wdl50);
    } catch (const std::exception& e) {
        std::snprintf(error, capacity, "%s", e.what());
        return -1;
    }
}
struct Ce5RootMove { int32_t dtz; uint8_t from; uint8_t to; uint8_t promotion; };

int ce5_tb_root_dtz(void* tables, const char* fen, unsigned ep, unsigned clock,
                       bool use_rule50, bool has_repeated,
                       Ce5RootMove* output, size_t output_capacity,
                       char* error, size_t capacity) noexcept {
    try {
        const Position position = Position::from_fen(fen);
        const auto moves = static_cast<Probe_Tables*>(tables)->probe_root_dtz(
            position, ep == 64 ? SQ_END : static_cast<Square>(ep), clock, use_rule50, has_repeated);
        if (moves.empty()) throw std::runtime_error("missing root DTZ probe");
        size_t count = 0;
        for (const auto& entry : moves) {
            if (count == output_capacity) throw std::runtime_error("move buffer too small");
            const Move move = entry.move;
            unsigned promotion = 0;
            if (move.is_promotion()) {
                switch (move.promotion()) {
                    case KNIGHT: promotion = 1; break;
                    case BISHOP: promotion = 2; break;
                    case ROOK: promotion = 3; break;
                    case QUEEN: promotion = 4; break;
                    default: throw std::runtime_error("invalid promotion");
                }
            }
            output[count++] = Ce5RootMove{entry.dtz, static_cast<uint8_t>(move.from()),
                static_cast<uint8_t>(move.to()), static_cast<uint8_t>(promotion)};
        }
        return static_cast<int>(count);
    } catch (const std::exception& e) {
        std::snprintf(error, capacity, "%s", e.what());
        return -1;
    }
}

}
