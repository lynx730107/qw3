"""Cache cooperative input coordinates outside K and unroll bounded input slots."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.register_iq4_cached'
out.mkdir(exist_ok=True)
kernel = (root / 'tests/.register_iq4/kernel.metal').read_text()
anchor = '''    auto cA = mm.get_right_input_cooperative_tensor<float, half, float>();
    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {'''
assert kernel.count(anchor) == 1
kernel = kernel.replace(anchor, '''    auto cA = mm.get_right_input_cooperative_tensor<float, half, float>();
    // Bound the private coordinate cache; reject a larger hardware layout.
    // Inactive slots stay masked, including any padding in the input layout.
    if (cA.get_capacity() > 128u) return;
    ushort packed_index[128];
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 128u; ++i) {
        if (i < cA.get_capacity() && cA.is_valid_element(i)) {
            auto index = cA.get_multidimensional_index(i);
            packed_index[i] = ushort(uint(index[1]) * 32u + uint(index[0]));
        } else packed_index[i] = 0xffffu;
    }
    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {''')
anchor = '''        for (uint16_t i = 0; i < cA.get_capacity(); ++i) {
            if (cA.is_valid_element(i)) {
                auto index = cA.get_multidimensional_index(i);
                const uint k = loop_k + uint(index[0]);
                const uint row = uint(r0 + index[1]);'''
assert kernel.count(anchor) == 1
kernel = kernel.replace(anchor, '''#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < 128u; ++i) {
            const ushort index = packed_index[i];
            if (index != 0xffffu) {
                const uint k = loop_k + uint(index & 31u);
                const uint row = uint(r0) + uint(index >> 5u);''')
(out / 'kernel.metal').write_text(kernel)
adapter = (root / 'tests/register_iq4.inc').read_text()
adapter = adapter.replace('"../qw3_metal.m"', '"../../qw3_metal.m"')
adapter = adapter.replace('"../qw3.c"', '"../../qw3.c"')
adapter = adapter.replace('@"tests/.register_iq4/kernel.metal"', '@"tests/.register_iq4_cached/kernel.metal"')
(out / 'adapter.inc').write_text(adapter)
driver = (root / 'tests/test_register_iq4.m').read_text()
driver = driver.replace('"register_iq4.inc"', '"adapter.inc"')
driver = driver.replace('order[]={0,1,0,1,0,1,0}', 'order[]={0,1,0,1,0}')
driver = driver.replace('block<7', 'block<5')
(out / 'driver.inc').write_text(driver)
