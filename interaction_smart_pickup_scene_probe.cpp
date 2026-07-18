#include "interaction_smart_pickup_scene.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr const char* kTargetSequenceId =
    "pickup_table__beer_10__001";
constexpr int32_t kTargetEntryLocalFrame = 101;
constexpr int32_t kTargetContactLocalFrame = 126;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void append_json_string(std::string& output, std::string_view value) {
    static constexpr char hexadecimal[] = "0123456789abcdef";
    output.push_back('"');
    for (unsigned char character : value) {
        switch (character) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (character < 0x20U) {
                output += "\\u00";
                output.push_back(hexadecimal[(character >> 4U) & 0x0fU]);
                output.push_back(hexadecimal[character & 0x0fU]);
            } else {
                require(
                    character < 0x80U,
                    "canonical evidence strings must be ASCII");
                output.push_back(static_cast<char>(character));
            }
        }
    }
    output.push_back('"');
}

class CanonicalJsonObject {
public:
    explicit CanonicalJsonObject(std::string& output) : output_(output) {
        output_.push_back('{');
    }

    void key(std::string_view value) {
        require(!finished_, "canonical JSON object is already closed");
        const std::string current(value);
        require(
            previous_key_.empty() || previous_key_ < current,
            "canonical JSON object keys are not strictly alphabetical");
        if (!first_) output_.push_back(',');
        append_json_string(output_, value);
        output_.push_back(':');
        previous_key_ = current;
        first_ = false;
    }

    void finish() {
        require(!finished_, "canonical JSON object was closed twice");
        output_.push_back('}');
        finished_ = true;
    }

private:
    std::string& output_;
    std::string previous_key_;
    bool first_ = true;
    bool finished_ = false;
};

std::string float32_hex(float value) {
    require(std::isfinite(value), "float32 evidence must be finite");
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));

    static constexpr char hexadecimal[] = "0123456789abcdef";
    std::string output = "0x00000000";
    for (size_t index = 0U; index < 8U; ++index) {
        const uint32_t shift = static_cast<uint32_t>((7U - index) * 4U);
        output[index + 2U] = hexadecimal[(bits >> shift) & 0x0fU];
    }
    return output;
}

void append_target_scalar(
    CanonicalJsonObject& scalars,
    std::string& output,
    std::string_view name,
    float value) {
    scalars.key(name);
    append_json_string(output, float32_hex(value));
}

void append_target_record(
    std::string& output,
    const interaction::InteractionTarget& target) {
    require(
        target.affordances.size() == 1U,
        "compiled smart-pickup target must have one affordance");
    const interaction::GraspAffordance& affordance =
        target.affordances.front();

    CanonicalJsonObject object(output);
    object.key("contact_local_frame");
    output += std::to_string(kTargetContactLocalFrame);
    object.key("entry_local_frame");
    output += std::to_string(kTargetEntryLocalFrame);
    object.key("scalars_f32_hex");
    {
        CanonicalJsonObject scalars(output);
        append_target_scalar(
            scalars, output, "approach_direction_object_x",
            affordance.approach_direction_object.x);
        append_target_scalar(
            scalars, output, "approach_direction_object_y",
            affordance.approach_direction_object.y);
        append_target_scalar(
            scalars, output, "approach_direction_object_z",
            affordance.approach_direction_object.z);
        append_target_scalar(
            scalars, output, "clearance_radius_m",
            affordance.clearance_radius);
        append_target_scalar(
            scalars, output, "grasp_position_object_x",
            affordance.hand_in_object.position.x);
        append_target_scalar(
            scalars, output, "grasp_position_object_y",
            affordance.hand_in_object.position.y);
        append_target_scalar(
            scalars, output, "grasp_position_object_z",
            affordance.hand_in_object.position.z);
        append_target_scalar(
            scalars, output, "grasp_rotation_object_w",
            affordance.hand_in_object.rotation.w);
        append_target_scalar(
            scalars, output, "grasp_rotation_object_x",
            affordance.hand_in_object.rotation.x);
        append_target_scalar(
            scalars, output, "grasp_rotation_object_y",
            affordance.hand_in_object.rotation.y);
        append_target_scalar(
            scalars, output, "grasp_rotation_object_z",
            affordance.hand_in_object.rotation.z);
        append_target_scalar(
            scalars, output, "object_dimensions_x",
            target.object_dimensions.x);
        append_target_scalar(
            scalars, output, "object_dimensions_y",
            target.object_dimensions.y);
        append_target_scalar(
            scalars, output, "object_dimensions_z",
            target.object_dimensions.z);
        append_target_scalar(
            scalars, output, "object_position_x",
            target.object_world.position.x);
        append_target_scalar(
            scalars, output, "object_position_y",
            target.object_world.position.y);
        append_target_scalar(
            scalars, output, "object_position_z",
            target.object_world.position.z);
        append_target_scalar(
            scalars, output, "object_rotation_w",
            target.object_world.rotation.w);
        append_target_scalar(
            scalars, output, "object_rotation_x",
            target.object_world.rotation.x);
        append_target_scalar(
            scalars, output, "object_rotation_y",
            target.object_world.rotation.y);
        append_target_scalar(
            scalars, output, "object_rotation_z",
            target.object_world.rotation.z);
        append_target_scalar(
            scalars, output, "table_position_x",
            target.table_world.position.x);
        append_target_scalar(
            scalars, output, "table_position_y",
            target.table_world.position.y);
        append_target_scalar(
            scalars, output, "table_position_z",
            target.table_world.position.z);
        append_target_scalar(
            scalars, output, "table_rotation_w",
            target.table_world.rotation.w);
        append_target_scalar(
            scalars, output, "table_rotation_x",
            target.table_world.rotation.x);
        append_target_scalar(
            scalars, output, "table_rotation_y",
            target.table_world.rotation.y);
        append_target_scalar(
            scalars, output, "table_rotation_z",
            target.table_world.rotation.z);
        append_target_scalar(
            scalars, output, "table_size_x", target.table_size.x);
        append_target_scalar(
            scalars, output, "table_size_y", target.table_size.y);
        append_target_scalar(
            scalars, output, "table_size_z", target.table_size.z);
        scalars.finish();
    }
    object.key("sequence_id");
    append_json_string(output, kTargetSequenceId);
    object.finish();
}

void append_slot_record(
    std::string& output,
    const interaction::GraspInteractionSlot& slot,
    const interaction::SmartPickupSlotProvenance& provenance) {
    require(slot.id != 0U, "compiled smart-pickup slot ID must be nonzero");
    require(
        slot.id == provenance.slot_id,
        "compiled smart-pickup slot and provenance IDs differ");
    require(
        provenance.sequence_id != nullptr &&
            provenance.sequence_id[0] != '\0',
        "compiled smart-pickup provenance sequence ID is empty");
    require(
        provenance.entry_local_frame >= 0 &&
            provenance.entry_local_frame < provenance.contact_local_frame,
        "compiled smart-pickup provenance event frames are invalid");

    CanonicalJsonObject object(output);
    object.key("contact_local_frame");
    output += std::to_string(provenance.contact_local_frame);
    object.key("entry_local_frame");
    output += std::to_string(provenance.entry_local_frame);
    object.key("prospective_root_x_object_f32_hex");
    append_json_string(output, float32_hex(slot.root_x_object_m));
    object.key("prospective_root_yaw_object_f32_hex");
    append_json_string(output, float32_hex(slot.root_yaw_object_radians));
    object.key("prospective_root_z_object_f32_hex");
    append_json_string(output, float32_hex(slot.root_z_object_m));
    object.key("sequence_id");
    append_json_string(output, provenance.sequence_id);
    object.key("slot_id");
    output += std::to_string(slot.id);
    object.finish();
}

std::string serialize_compiled_scene() {
    const interaction::InteractionTarget target =
        interaction::make_smart_pickup_demo_target();
    require(
        target.affordances.size() == 1U,
        "compiled smart-pickup target must have one affordance");
    const auto& slots = target.affordances.front().interaction_slots;
    const auto& provenance =
        interaction::smart_pickup_demo_slot_provenance();
    require(
        slots.size() == provenance.size() && slots.size() == 3U,
        "compiled smart-pickup scene must contain exactly three joined slots");

    std::string output;
    output.reserve(2048U);
    CanonicalJsonObject object(output);
    object.key("record_type");
    append_json_string(output, "compiled_scene");
    object.key("slots");
    output.push_back('[');
    for (size_t index = 0U; index < slots.size(); ++index) {
        if (index != 0U) output.push_back(',');
        append_slot_record(output, slots[index], provenance[index]);
    }
    output.push_back(']');
    object.key("target");
    append_target_record(output, target);
    object.finish();
    return output;
}

int run(int argc, char** argv) {
    if (argc != 2 || std::string_view(argv[1]) != "--json") {
        std::cerr << "usage: interaction_smart_pickup_scene_probe --json\n";
        return 2;
    }
    const std::string output = serialize_compiled_scene();
    std::cout << output << '\n';
    if (!std::cout) fail("could not write compiled scene evidence");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "interaction_smart_pickup_scene_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
