const textBox = document.getElementById("textBox");
const keyboard = document.getElementById("keyboard");
const suggestions = document.getElementById("suggestions");
const clearBtn = document.getElementById("clearBtn");

const rows = [
  ["q","w","e","r","t","y","u","i","o","p"],
  ["a","s","d","f","g","h","j","k","l"],
  ["z","x","c","v","b","n","m"]
];

function makeKey(label, action="", extra="") {
  const b = document.createElement("button");
  b.className = "key " + extra;
  b.textContent = label;
  b.addEventListener("click", () => {
    if (action === "backspace") {
      textBox.value = textBox.value.slice(0, -1);
    } else if (action === "space") {
      textBox.value += " ";
    } else if (action === "enter") {
      textBox.value += "\n";
    } else {
      textBox.value += label;
    }
    textBox.focus();
    updateSuggestions();
  });
  return b;
}

rows.forEach((letters, i) => {
  const row = document.createElement("div");
  row.className = "row";
  letters.forEach(letter => row.appendChild(makeKey(letter)));
  keyboard.appendChild(row);
});

const bottom = document.createElement("div");
bottom.className = "row";
bottom.appendChild(makeKey("⌫", "backspace", "wide"));
bottom.appendChild(makeKey("Space", "space", "space"));
bottom.appendChild(makeKey("↵", "enter", "wide"));
keyboard.appendChild(bottom);

async function updateSuggestions() {
  const text = textBox.value;
  if (!text.trim()) {
    suggestions.innerHTML = '<button class="suggestion muted">Suggestions appear here</button>';
    return;
  }

  try {
    const response = await fetch("/suggest", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });
    const data = await response.json();
    suggestions.innerHTML = "";

    const words = data.suggestions || [];
    if (!words.length) {
      suggestions.innerHTML = '<button class="suggestion muted">No corpus match</button>';
      return;
    }

    words.forEach(word => {
      const b = document.createElement("button");
      b.className = "suggestion";
      b.textContent = word;
      b.onclick = () => insertSuggestion(word);
      suggestions.appendChild(b);
    });
  } catch (err) {
    suggestions.innerHTML = '<button class="suggestion muted">Server unavailable</button>';
  }
}

function insertSuggestion(word) {
  const current = textBox.value;
  const match = current.match(/([a-z]+)$/i);
  if (match) {
    textBox.value = current.slice(0, match.index) + word + " ";
  } else {
    textBox.value += word + " ";
  }
  textBox.focus();
  updateSuggestions();
}

textBox.addEventListener("input", updateSuggestions);
clearBtn.addEventListener("click", () => {
  textBox.value = "";
  textBox.focus();
  updateSuggestions();
});

textBox.focus();
