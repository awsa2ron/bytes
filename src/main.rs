use bytes::{Buf, rBytes};

fn main() {
let mut mem = rBytes::from("Hello world");
let a = mem.slice(0..5);

// assert_eq!(a, "Hello");
println!("{:?}", a);

let b = mem.split_to(6);

println!("{:?}", b);
// assert_eq!(mem, "world");
// assert_eq!(b, "Hello ");

println!("{:?}", mem);
mem.advance(1);
println!("{:?}", mem);
mem.advance(1);
println!("{:?}", mem);
mem.advance(1);
println!("{:?}", mem);
println!("{:?}", mem.chunk());
println!("{:?}", mem.remaining());
println!("{:?}", mem.get_u8());

}
