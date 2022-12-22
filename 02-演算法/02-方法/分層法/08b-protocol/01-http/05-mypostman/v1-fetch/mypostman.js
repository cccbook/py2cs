console.log('args=', Deno.args)

const textResponse = await fetch(Deno.args[0]);
const textData = await textResponse.text();
console.log(textData);