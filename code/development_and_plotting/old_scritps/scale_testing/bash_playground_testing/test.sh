#!/bin/bash

# Liste mit Werten
values=(32 72 144 216)

# Schleife über alle Werte
for value in "${values[@]}"; do
    echo "Bearbeite Wert: $value"

    if [ "$value" -le 72 ]; then
        echo "Fall A: $value ist kleinergleich als 72"
        # Hier kannst du Aktion A ausführen
    else
        # Modulo 72 berechnen
        mod=$(( value / 72 ))
        echo "Fall B: $value ist größer 72"
        echo "Modulo 72 ergibt: $mod"
        # Hier kannst du Aktion B ausführen
    fi

    echo "---"
done