import json
import pygame
import sys

# Importing custom modules
import animator
import background
import logic
import sprites
import synchronizer
from constants import *

# ----------------------------------------------
# Sequence Loading
# ----------------------------------------------

# Function to load configuration from JSON file
def load_configuration(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)

# Main game loop
def main(initial_state_file, sequence_file):
    # Load initial and sequence states
    initial_state = load_configuration(f"../{initial_state_file}")
    sequence = load_configuration(f"../{sequence_file}")

    # These two variables are important for the animator and the sequencer
    number_of_disks = sprites.obtain_number_of_disks(initial_state)
    disk_height = sprites.obtain_disks_height(number_of_disks)

    # ----------------------------------------------
    # Pygame Initialization
    # ----------------------------------------------

    # Initialize Pygame and create the display screen
    screen = initialize_pygame()
    clock = pygame.time.Clock()
    pygame.display.set_caption("Hanoi's tower simulation")

    # Initialize the logic (registers how many disks are in each peg and the height of the tower per each peg)
    hanoi_base = logic.initialize_logic(initial_state, disk_height)

    # Initialize the disk sprites
    disks_sprites_groups = pygame.sprite.Group()
    disks_sprites = sprites.create_sprites(number_of_disks, disk_height, hanoi_base, initial_state)
    for disk_id in disks_sprites:
        disks_sprites_groups.add(disks_sprites[disk_id])

    # Initiate the synchronizer
    sync_manager = synchronizer.Synchronizer(sequence)
    # Initiate the animator
    anim_manager = animator.Animator(hanoi_base, disk_height)

    # Flag to execute the next sequence
    flag_execute_next_seq = anim_manager.ask_new_seq

    while True:  # Sequence Loop
        handle_events()  # Handle Pygame events

        # Synchronize
        if flag_execute_next_seq:
            seq = sync_manager.update()  # Get the next sequence from the synchronizer
            anim_manager.get_sequence(seq)  # Pass the sequence to the animator

        # Animation
        anim_manager.animate(disks_sprites)  # Animate the disks
        disks_sprites_groups.update()  # Update the disk sprites
        flag_execute_next_seq = anim_manager.ask_new_seq  # Check if there's a new sequence to execute

        # Draw all elements
        background.draw_background(screen)  # Draw the background
        disks_sprites_groups.draw(screen)  # Draw the disk sprites

        pygame.display.flip()  # Update the display

        # Limit the FPS by sleeping for the remainder of the frame time
        clock.tick(FPS)

# Initialize Pygame
def initialize_pygame():
    pygame.init()
    return pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Handle Pygame events
def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# -----------
if __name__ == "__main__":
    ## Valores por defecto
    default_initial_state = "initial_state.json"
    default_sequence = "sequence.json"
    
    # Asignar archivos según los argumentos (si se proporcionan)
    if len(sys.argv) == 3:  # Si se pasan ambos archivos
        initial_state_file = sys.argv[1]
        sequence_file = sys.argv[2]
    elif len(sys.argv) == 2:  # Si solo se pasa un archivo (podría ser opcional)
        print("Advertencia: Se esperaban 2 archivos (initial_state y sequence). Usando valores por defecto para el segundo.")
        initial_state_file = sys.argv[1]
        sequence_file = default_sequence
    else:  # Si no se pasan argumentos, usar valores por defecto
        print(f"Advertencia: No se especificaron archivos. Usando '{default_initial_state}' y '{default_sequence}' por defecto.")
        initial_state_file = default_initial_state
        sequence_file = default_sequence
    
    main(initial_state_file, sequence_file)