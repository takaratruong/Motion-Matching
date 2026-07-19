#include "interaction_pickup_provenance.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace interaction {
namespace {

bool valid_range(int32_t range_start, int32_t range_stop) {
    return range_start >= 0 && range_start < range_stop;
}

uint32_t float_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool same_float_bits(float left, float right) {
    return float_bits(left) == float_bits(right);
}

bool same_vec3_bits(vec3 left, vec3 right) {
    return same_float_bits(left.x, right.x) &&
        same_float_bits(left.y, right.y) &&
        same_float_bits(left.z, right.z);
}

bool same_quat_bits(quat left, quat right) {
    return same_float_bits(left.w, right.w) &&
        same_float_bits(left.x, right.x) &&
        same_float_bits(left.y, right.y) &&
        same_float_bits(left.z, right.z);
}

bool same_provenance(
    const PickupSourceProvenance& left,
    const PickupSourceProvenance& right) {
    return left.clip_ordinal == right.clip_ordinal &&
        left.range_start == right.range_start &&
        left.range_stop == right.range_stop &&
        left.entry_global_frame == right.entry_global_frame &&
        left.contact_global_frame == right.contact_global_frame &&
        left.lift_global_frame == right.lift_global_frame &&
        left.hold_global_frame == right.hold_global_frame &&
        left.active_hand == right.active_hand &&
        left.object_profile_id == right.object_profile_id &&
        same_vec3_bits(
            left.object_bounds.center_object,
            right.object_bounds.center_object) &&
        same_vec3_bits(
            left.object_bounds.half_extents_object,
            right.object_bounds.half_extents_object) &&
        same_vec3_bits(
            left.hand_in_object.position,
            right.hand_in_object.position) &&
        same_quat_bits(
            left.hand_in_object.rotation,
            right.hand_in_object.rotation) &&
        same_float_bits(
            left.source_support_height_m,
            right.source_support_height_m);
}

bool finite_vec3(vec3 value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z);
}

bool finite_quat(quat value) {
    return std::isfinite(value.w) &&
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z);
}

bool valid_raw_provenance(const PickupSourceProvenance& value) {
    const uint8_t hand = static_cast<uint8_t>(value.active_hand);
    const double rotation_norm =
        static_cast<double>(value.hand_in_object.rotation.w) *
            value.hand_in_object.rotation.w +
        static_cast<double>(value.hand_in_object.rotation.x) *
            value.hand_in_object.rotation.x +
        static_cast<double>(value.hand_in_object.rotation.y) *
            value.hand_in_object.rotation.y +
        static_cast<double>(value.hand_in_object.rotation.z) *
            value.hand_in_object.rotation.z;
    return value.clip_ordinal >= 0 &&
        valid_range(value.range_start, value.range_stop) &&
        value.range_start <= value.entry_global_frame &&
        value.entry_global_frame < value.contact_global_frame &&
        value.contact_global_frame < value.lift_global_frame &&
        value.lift_global_frame < value.hold_global_frame &&
        value.hold_global_frame < value.range_stop &&
        hand <= static_cast<uint8_t>(Hand::Right) &&
        value.object_profile_id != 0U &&
        finite_vec3(value.object_bounds.center_object) &&
        finite_vec3(value.object_bounds.half_extents_object) &&
        value.object_bounds.half_extents_object.x > 0.0F &&
        value.object_bounds.half_extents_object.y > 0.0F &&
        value.object_bounds.half_extents_object.z > 0.0F &&
        finite_vec3(value.hand_in_object.position) &&
        finite_quat(value.hand_in_object.rotation) &&
        std::isfinite(rotation_norm) && rotation_norm > 0.0 &&
        std::isfinite(value.source_support_height_m);
}

int hexadecimal_value(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

std::optional<uint64_t> source_id_from_join_sha(
    const std::string& value) {
    if (value.size() != 64U) return std::nullopt;
    uint64_t source_id = 0U;
    for (size_t index = 0U; index < value.size(); ++index) {
        const int nibble = hexadecimal_value(value[index]);
        if (nibble < 0) return std::nullopt;
        if (index < 16U) {
            source_id = (source_id << 4U) |
                static_cast<uint64_t>(nibble);
        }
    }
    return source_id;
}

void validate_certified_row(const CertifiedPickupSourceIdentity& row) {
    if (!valid_raw_provenance(row.provenance) ||
        row.sequence_id.empty() || row.object_id.empty() ||
        row.reverse_start_global_frame <
            row.provenance.hold_global_frame ||
        row.reverse_start_global_frame >= row.provenance.range_stop) {
        throw std::invalid_argument(
            "invalid certified pickup source metadata");
    }
    const std::optional<uint64_t> expected =
        source_id_from_join_sha(row.join_key_sha256);
    if (!expected.has_value() || *expected == 0U ||
        row.source_id != *expected) {
        throw std::invalid_argument(
            "invalid certified pickup source join identity");
    }
}

}  // namespace

int32_t checked_pickup_global_to_local_frame(
    int32_t global_frame,
    int32_t range_start,
    int32_t range_stop) {
    if (!valid_range(range_start, range_stop)) {
        throw std::invalid_argument("invalid pickup frame range");
    }
    if (global_frame < range_start || global_frame >= range_stop) {
        throw std::out_of_range("pickup global frame is outside its range");
    }
    return global_frame - range_start;
}

int32_t checked_pickup_local_to_global_frame(
    int32_t local_frame,
    int32_t range_start,
    int32_t range_stop) {
    if (!valid_range(range_start, range_stop)) {
        throw std::invalid_argument("invalid pickup frame range");
    }
    if (local_frame < 0) {
        throw std::out_of_range("pickup local frame is negative");
    }
    const int64_t global = static_cast<int64_t>(range_start) +
        static_cast<int64_t>(local_frame);
    if (global > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("pickup global frame addition overflow");
    }
    const int32_t clip_length = range_stop - range_start;
    if (local_frame >= clip_length) {
        throw std::out_of_range("pickup local frame is outside its range");
    }
    return static_cast<int32_t>(global);
}

CertifiedPickupSourceIdentity make_certified_pickup_source_identity(
    const CertifiedPickupSourceLocalRow& row) {
    if (!(row.entry_local_frame < row.contact_local_frame &&
          row.contact_local_frame < row.lift_local_frame &&
          row.lift_local_frame < row.hold_local_frame &&
          row.hold_local_frame <= row.reverse_start_local_frame)) {
        throw std::invalid_argument(
            "certified pickup local events are unordered");
    }
    CertifiedPickupSourceIdentity result{};
    result.provenance.clip_ordinal = row.clip_ordinal;
    result.provenance.range_start = row.range_start;
    result.provenance.range_stop = row.range_stop;
    result.provenance.entry_global_frame =
        checked_pickup_local_to_global_frame(
            row.entry_local_frame, row.range_start, row.range_stop);
    result.provenance.contact_global_frame =
        checked_pickup_local_to_global_frame(
            row.contact_local_frame, row.range_start, row.range_stop);
    result.provenance.lift_global_frame =
        checked_pickup_local_to_global_frame(
            row.lift_local_frame, row.range_start, row.range_stop);
    result.provenance.hold_global_frame =
        checked_pickup_local_to_global_frame(
            row.hold_local_frame, row.range_start, row.range_stop);
    result.reverse_start_global_frame =
        checked_pickup_local_to_global_frame(
            row.reverse_start_local_frame, row.range_start, row.range_stop);
    result.provenance.active_hand = row.active_hand;
    result.provenance.object_profile_id = row.object_profile_id;
    result.provenance.object_bounds = row.object_bounds;
    result.provenance.hand_in_object = row.hand_in_object;
    result.provenance.source_support_height_m =
        row.source_support_height_m;
    result.sequence_id = row.sequence_id;
    result.object_id = row.object_id;
    result.join_key_sha256 = row.join_key_sha256;
    result.source_id = row.source_id;
    return result;
}

CertifiedPickupSourceRegistry::CertifiedPickupSourceRegistry(
    std::vector<CertifiedPickupSourceIdentity> rows)
    : rows_(std::move(rows)) {
    for (size_t index = 0U; index < rows_.size(); ++index) {
        validate_certified_row(rows_[index]);
        for (size_t earlier = 0U; earlier < index; ++earlier) {
            if (same_provenance(
                    rows_[earlier].provenance,
                    rows_[index].provenance)) {
                throw std::invalid_argument(
                    "duplicate certified pickup source provenance");
            }
        }
    }
}

std::optional<CertifiedPickupSourceIdentity>
CertifiedPickupSourceRegistry::resolve_exact(
    const PickupSourceProvenance& actual) const {
    for (const CertifiedPickupSourceIdentity& row : rows_) {
        if (same_provenance(row.provenance, actual)) return row;
    }
    return std::nullopt;
}

}  // namespace interaction
