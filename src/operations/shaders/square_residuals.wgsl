@group(0) @binding(0)
var<storage, read> input: array<#INPUT_TYPE>;

@group(0) @binding(1)
var<storage, read_write> output: array<#INPUT_TYPE>;

@group(0) @binding(2)
var<uniform> mean: f32;

struct PushConstants{
    total_elements: u32,
}

var<push_constant> pc: PushConstants;

#import get_flat_idx

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(
	@builtin(workgroup_id) wg_id: vec3<u32>,
	@builtin(num_workgroups) wg_num: vec3<u32>,
	@builtin(local_invocation_index) local_index: u32,
){
	let flat_idx = get_flat_idx(wg_id, wg_num, local_index);
    if (flat_idx >= pc.total_elements) {
        return;
    }

    let datum = input[flat_idx];
	#if NANPROTECT{
        if (bitcast<u32>(datum) != 4294967295u){
            let diff = datum - mean;
            output[flat_idx] = diff * diff;
        } else {
            output[flat_idx] = 0.0;
        }
    } #else {
        let diff = datum - mean;
        output[flat_idx] = diff * diff;
    }
}
