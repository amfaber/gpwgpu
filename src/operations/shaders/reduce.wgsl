struct PushConstants{
    stride: u32,
    length: u32,
}

#ifdef OUTPLACE
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

#else
@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

#endif

var<push_constant> pc: PushConstants;

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
    if (global_id.x >= pc.stride) {
        return;
    }

    var idx = global_id.x;
    let stride = pc.stride;
    
    var datum = input[idx];
    var acc = datum;
    
    #for I in 2u..UNROLL
    idx = global_id.x + stride * (#I - 1u);
    datum = input[idx];
    #if NANPROTECT == true
    if (bitcast<u32>(datum) != 4294967295u){
        #OPERATION
    }
    #else
    #OPERATION
    #endif
    #endfor

    idx = global_id.x + stride * (#UNROLL - 1u);
    datum = input[idx];
    
    if (idx < pc.length)
    #if NANPROTECT == true
    & (bitcast<u32>(datum) != 4294967295u
    #endif
    {
        #OPERATION
    }

    #ifdef OUTPLACE
    output[global_id.x] = acc;
    #else
    input[global_id.x] = acc;
    #endif
}
