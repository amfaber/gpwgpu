struct PushConstants{
    stride: u32,
    length: u32,
}

#if OUTPLACE{
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

} #else {
@group(0) @binding(0)
var<storage, read_write> input: array<f32>;
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
	var idx = flat_idx;
    if (idx >= pc.stride) {
        return;
    }

    let stride = pc.stride;
    
    var datum = input[idx];
    var acc = datum;
    
    #for I in 1..UNROLL-1{
        idx = flat_idx + stride * u32(#I);
        datum = input[idx];
        #if NANPROTECT{
            if (bitcast<u32>(datum) != 4294967295u){
                #OPERATION
            }
        } #else {
            #OPERATION
        }
    }

    idx = flat_idx + stride * (#UNROLL - 1u);
    datum = input[idx];
    
    if (idx < pc.length)
    #if NANPROTECT{
        & (bitcast<u32>(datum) != 4294967295u)
    }
    {
        #OPERATION
    }

    #if OUTPLACE{
        output[flat_idx] = acc;
    } #else {
        input[flat_idx] = acc;
    }
}
