document.getElementById("predictBtn").addEventListener("click", async () => {
  const raw = document.getElementById("series").value;
  const series = raw
    .split(",")
    .map((x) => Number(x.trim()))
    .filter((x) => !Number.isNaN(x));

  const output = document.getElementById("output");
  output.textContent = "Running...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ series }),
    });

    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    output.textContent = `Error: ${err}`;
  }
});