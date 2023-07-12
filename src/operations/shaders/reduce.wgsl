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

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
    if (global_id.x >= pc.stride) {
        return;
    }

    var idx = global_id.x;
    let stride = pc.stride;
    
    var datum = input[idx];
    var acc = datum;
    
    #for I in 1..UNROLL-1{
        idx = global_id.x + stride * u32(#I);
        datum = input[idx];
        #if NANPROTECT{
            if (bitcast<u32>(datum) != 4294967295u){
                #OPERATION
            }
        } #else {
            #OPERATION
        }
    }

    idx = global_id.x + stride * (#UNROLL - 1u);
    datum = input[idx];
    
    if (idx < pc.length)
    #if NANPROTECT{
        & (bitcast<u32>(datum) != 4294967295u
    }
    {
        #OPERATION
    }

    #if OUTPLACE{
        output[global_id.x] = acc;
    } #else {
        input[global_id.x] = acc;
    }
}
