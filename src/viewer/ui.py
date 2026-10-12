import pygame
from typing import Callable, Any


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Safe font loader with fallback."""
    try:
        return pygame.font.SysFont("Helvetica", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)


class UIButton:
    """Clickable UI button with hover highlight and shortcut badge."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        callback: Callable[[], Any],
        shortcut: str = "",
        tooltip: str = "",
        accent_color: tuple[int, int, int] | None = None,
    ):
        self.rect = rect
        self.label = label
        self.callback = callback
        self.shortcut = shortcut
        self.tooltip = tooltip
        self.accent_color = accent_color
        self.is_hovered = False

    def handle_mouse_motion(self, pos: tuple[int, int]) -> bool:
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def handle_click(self, pos: tuple[int, int]) -> bool:
        if self.rect.collidepoint(pos):
            self.callback()
            return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        # Background
        if self.is_hovered:
            bg_color = (48, 68, 96)
            border_color = self.accent_color if self.accent_color else (90, 130, 185)
        else:
            bg_color = (25, 35, 52)
            border_color = (45, 62, 88)

        pygame.draw.rect(screen, bg_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, border_color, self.rect, width=1, border_radius=5)

        # Content layout
        mid_y = self.rect.centery
        x_cursor = self.rect.x + 8

        if self.shortcut:
            # Small pill for shortcut
            sc_surf = small_font.render(self.shortcut, True, (160, 190, 225))
            sc_w = sc_surf.get_width() + 8
            sc_rect = pygame.Rect(x_cursor, mid_y - 8, sc_w, 16)
            pygame.draw.rect(screen, (36, 48, 70), sc_rect, border_radius=3)
            screen.blit(sc_surf, (sc_rect.x + 4, sc_rect.y + 1))
            x_cursor += sc_w + 6

        # Label
        lbl_surf = small_font.render(self.label, True, (230, 240, 250) if not self.accent_color else self.accent_color)
        screen.blit(lbl_surf, (x_cursor, mid_y - lbl_surf.get_height() // 2))


class UIToggleBadge:
    """
    Sleek restructured toggle element.
    Displays shortcut key, feature name, and clear ON/OFF status badge with color coding.
    Clickable directly with mouse or triggered via keypress.
    """

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        shortcut: str,
        get_state: Callable[[], bool],
        on_toggle: Callable[[], None],
    ):
        self.rect = rect
        self.label = label
        self.shortcut = shortcut
        self.get_state = get_state
        self.on_toggle = on_toggle
        self.is_hovered = False

    def handle_mouse_motion(self, pos: tuple[int, int]) -> bool:
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def handle_click(self, pos: tuple[int, int]) -> bool:
        if self.rect.collidepoint(pos):
            self.on_toggle()
            return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        is_on = self.get_state()

        # Box colors based on state and hover
        if is_on:
            if self.is_hovered:
                bg_col = (24, 58, 50)
                border_col = (60, 190, 140)
            else:
                bg_col = (18, 44, 38)
                border_col = (40, 140, 100)
            status_text = "ON"
            status_color = (65, 235, 165)
            badge_bg = (24, 75, 55)
        else:
            if self.is_hovered:
                bg_col = (36, 46, 62)
                border_col = (75, 95, 125)
            else:
                bg_col = (22, 28, 40)
                border_col = (40, 52, 70)
            status_text = "OFF"
            status_color = (130, 145, 165)
            badge_bg = (32, 40, 54)

        pygame.draw.rect(screen, bg_col, self.rect, border_radius=5)
        pygame.draw.rect(screen, border_col, self.rect, width=1, border_radius=5)

        mid_y = self.rect.centery

        # 1. Shortcut pill
        sc_surf = small_font.render(self.shortcut, True, (170, 195, 220))
        sc_w = sc_surf.get_width() + 6
        sc_rect = pygame.Rect(self.rect.x + 6, mid_y - 8, sc_w, 16)
        pygame.draw.rect(screen, (34, 45, 64), sc_rect, border_radius=3)
        screen.blit(sc_surf, (sc_rect.x + 3, sc_rect.y + 1))

        # 2. Toggle Label
        lbl_surf = small_font.render(self.label, True, (225, 235, 245) if is_on else (160, 175, 195))
        screen.blit(lbl_surf, (self.rect.x + sc_w + 12, mid_y - lbl_surf.get_height() // 2))

        # 3. Status Pill on right
        stat_surf = small_font.render(status_text, True, status_color)
        pill_w = stat_surf.get_width() + 10
        pill_rect = pygame.Rect(self.rect.right - pill_w - 6, mid_y - 8, pill_w, 16)
        pygame.draw.rect(screen, badge_bg, pill_rect, border_radius=3)
        pygame.draw.rect(screen, border_col, pill_rect, width=1, border_radius=3)
        screen.blit(stat_surf, (pill_rect.x + 5, pill_rect.y + 1))


class UIStepper:
    """
    Live parameter controller with [-] and [+] buttons and current value readout.
    Allows changing key stats live rather than in code.
    """

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        get_val: Callable[[], Any],
        on_dec: Callable[[], None],
        on_inc: Callable[[], None],
        formatter: Callable[[Any], str] = str,
    ):
        self.rect = rect
        self.label = label
        self.get_val = get_val
        self.on_dec = on_dec
        self.on_inc = on_inc
        self.formatter = formatter

        btn_w = 24
        btn_h = 22
        mid_y = rect.centery
        self.btn_dec_rect = pygame.Rect(rect.right - (btn_w * 2 + 76), mid_y - btn_h // 2, btn_w, btn_h)
        self.val_rect = pygame.Rect(self.btn_dec_rect.right + 2, mid_y - btn_h // 2, 70, btn_h)
        self.btn_inc_rect = pygame.Rect(self.val_rect.right + 2, mid_y - btn_h // 2, btn_w, btn_h)

        self.hover_dec = False
        self.hover_inc = False

    def handle_mouse_motion(self, pos: tuple[int, int]) -> bool:
        self.hover_dec = self.btn_dec_rect.collidepoint(pos)
        self.hover_inc = self.btn_inc_rect.collidepoint(pos)
        return self.rect.collidepoint(pos)

    def handle_click(self, pos: tuple[int, int]) -> bool:
        if self.btn_dec_rect.collidepoint(pos):
            self.on_dec()
            return True
        elif self.btn_inc_rect.collidepoint(pos):
            self.on_inc()
            return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        # Row subtle background
        row_bg = (20, 26, 38)
        pygame.draw.rect(screen, row_bg, self.rect, border_radius=4)
        pygame.draw.rect(screen, (35, 46, 64), self.rect, width=1, border_radius=4)

        mid_y = self.rect.centery

        # Label on left
        lbl_surf = small_font.render(self.label, True, (195, 210, 228))
        screen.blit(lbl_surf, (self.rect.x + 8, mid_y - lbl_surf.get_height() // 2))

        # Decrement button
        dec_bg = (52, 72, 100) if self.hover_dec else (32, 44, 62)
        dec_border = (80, 120, 170) if self.hover_dec else (50, 70, 96)
        pygame.draw.rect(screen, dec_bg, self.btn_dec_rect, border_radius=3)
        pygame.draw.rect(screen, dec_border, self.btn_dec_rect, width=1, border_radius=3)
        dec_txt = font.render("-", True, (240, 245, 255))
        screen.blit(dec_txt, (self.btn_dec_rect.centerx - dec_txt.get_width() // 2, self.btn_dec_rect.centery - dec_txt.get_height() // 2 - 1))

        # Value display
        pygame.draw.rect(screen, (16, 21, 32), self.val_rect, border_radius=3)
        pygame.draw.rect(screen, (40, 52, 72), self.val_rect, width=1, border_radius=3)
        val_str = self.formatter(self.get_val())
        val_surf = small_font.render(val_str, True, (100, 210, 255))
        screen.blit(val_surf, (self.val_rect.centerx - val_surf.get_width() // 2, self.val_rect.centery - val_surf.get_height() // 2))

        # Increment button
        inc_bg = (52, 72, 100) if self.hover_inc else (32, 44, 62)
        inc_border = (80, 120, 170) if self.hover_inc else (50, 70, 96)
        pygame.draw.rect(screen, inc_bg, self.btn_inc_rect, border_radius=3)
        pygame.draw.rect(screen, inc_border, self.btn_inc_rect, width=1, border_radius=3)
        inc_txt = font.render("+", True, (240, 245, 255))
        screen.blit(inc_txt, (self.btn_inc_rect.centerx - inc_txt.get_width() // 2, self.btn_inc_rect.centery - inc_txt.get_height() // 2 - 1))


class UIManager:
    """
    Manages HUD panels, live stats controllers, restructured display toggles,
    and boundary legend with modern responsive layouts and mouse interaction.
    """

    def __init__(self, viewer: Any):
        self.viewer = viewer
        self.is_collapsed = False

        self.font = get_font(14)
        self.small_font = get_font(12)
        self.header_font = get_font(15, bold=True)
        self.title_font = get_font(16, bold=True)
        self.section_font = get_font(11, bold=True)

        self.panel_width = 300
        self.panel_rect = pygame.Rect(16, 16, self.panel_width, 620)
        self.collapse_btn_rect = pygame.Rect(self.panel_rect.right - 28, 22, 20, 20)
        self.hover_collapse = False

        self.steppers: list[UIStepper] = []
        self.toggles: list[UIToggleBadge] = []
        self.action_buttons: list[UIButton] = []
        self.simulation_buttons: list[UIButton] = []
        self.legend_rect = pygame.Rect(viewer.width - 386, viewer.height - 137, 370, 132)

        self._build_ui()

    def _build_ui(self):
        """Constructs UI widgets with clean hierarchical layout."""
        v = self.viewer
        x = self.panel_rect.x + 10
        w = self.panel_rect.width - 20

        # --- Key Stats Live Steppers ---
        self.steppers = [
            UIStepper(
                rect=pygame.Rect(x, 0, w, 28),
                label="Base Plates",
                get_val=lambda: v.plate_generator.base_num_plates if v.plate_generator else 16,
                on_dec=lambda: v.adjust_base_plates(-1),
                on_inc=lambda: v.adjust_base_plates(1),
                formatter=lambda val: f"{val} Plates",
            ),
            UIStepper(
                rect=pygame.Rect(x, 0, w, 28),
                label="Ocean Crust",
                get_val=lambda: v.plate_generator.oceanic_ratio if v.plate_generator else 0.70,
                on_dec=lambda: v.adjust_oceanic_ratio(-0.05),
                on_inc=lambda: v.adjust_oceanic_ratio(0.05),
                formatter=lambda val: f"{int(round(val * 100))}%",
            ),
            UIStepper(
                rect=pygame.Rect(x, 0, w, 28),
                label="Cont Land",
                get_val=lambda: v.plate_generator.continental_land_percent if v.plate_generator else 50,
                on_dec=lambda: v.adjust_continental_land_percent(-5),
                on_inc=lambda: v.adjust_continental_land_percent(5),
                formatter=lambda val: f"{val}%",
            ),
            UIStepper(
                rect=pygame.Rect(x, 0, w, 28),
                label="Mesh Subdiv",
                get_val=lambda: v.mesh.subdivisions,
                on_dec=lambda: v.adjust_subdivisions(-1),
                on_inc=lambda: v.adjust_subdivisions(1),
                formatter=lambda val: f"Level {val}",
            ),
            UIStepper(
                rect=pygame.Rect(x, 0, w, 28),
                label="Warp Jagg",
                get_val=lambda: v.plate_generator.warp_strength if v.plate_generator else 0.16,
                on_dec=lambda: v.adjust_warp_strength(-0.02),
                on_inc=lambda: v.adjust_warp_strength(0.02),
                formatter=lambda val: f"{val:.2f}",
            ),
            UIStepper(
                rect=pygame.Rect(x, 0, w, 28),
                label="Max Micro",
                get_val=lambda: v.plate_generator.max_microplates if v.plate_generator else 6,
                on_dec=lambda: v.adjust_max_microplates(-1),
                on_inc=lambda: v.adjust_max_microplates(1),
                formatter=lambda val: f"{val} Max",
            ),
            UIStepper(
                rect=pygame.Rect(x, 0, w, 28),
                label="Auto-Spin",
                get_val=lambda: v.rotation_speed,
                on_dec=lambda: v.adjust_rotation_speed(-0.2),
                on_inc=lambda: v.adjust_rotation_speed(0.2),
                formatter=lambda val: f"{val:.1f}x",
            ),
            UIStepper(
                rect=pygame.Rect(x, 0, w, 28),
                label="Sim Speed",
                get_val=lambda: v.simulation.playback_steps_per_second if v.simulation else 1.0,
                on_dec=lambda: v.adjust_simulation_speed(0.5),
                on_inc=lambda: v.adjust_simulation_speed(2.0),
                formatter=lambda val: f"{val:g} Myr/s",
            ),
        ]

        # --- Restructured Display Toggles (2 columns) ---
        col_w = (w - 6) // 2
        toggle_defs = [
            ("Boundaries", "B", lambda: v.show_boundaries, v.toggle_boundaries),
            ("Drift Arrows", "D", lambda: v.show_drift_arrows, v.toggle_drift_arrows),
            ("Wireframe", "W", lambda: v.show_wireframe, v.toggle_wireframe),
            ("Shading", "S", lambda: v.show_shading, v.toggle_shading),
            ("Pole Markers", "O", lambda: v.show_orientation_axes, v.toggle_orientation_axes),
            ("Adapt LOD", "L", lambda: v.use_adaptive_lod, v.toggle_adaptive_lod),
            ("Auto-Spin", "SPC", lambda: v.is_rotating_auto, v.toggle_auto_rotation),
            ("Invert Drag", "I", lambda: (v.camera.invert_x and v.camera.invert_y), v.toggle_invert_controls),
            ("Lithosphere B/W", "N", lambda: v.show_lithosphere, v.toggle_lithosphere),
            ("Terrain Colors", "M", lambda: v.show_terrain_colors, v.toggle_terrain_colors),
        ]

        self.toggles = []
        for label, shortcut, getter, toggler in toggle_defs:
            self.toggles.append(
                UIToggleBadge(
                    rect=pygame.Rect(0, 0, col_w, 26),
                    label=label,
                    shortcut=shortcut,
                    get_state=getter,
                    on_toggle=toggler,
                )
            )

        # --- Quick Action Buttons ---
        act_col_w = (w - 6) // 2
        action_defs = [
            ("New Plates", "G", v.action_new_plates, (80, 200, 255)),
            ("Fracture / Fuse", "F", v.action_fracture_microplate, (255, 180, 70)),
            ("Undo Fract", "U", v.action_undo_fracture, (130, 190, 255)),
            ("Reset Fract", "Z", v.action_reset_fractures, (220, 140, 140)),
            ("Reset Cam", "R", v.action_reset_camera, (180, 190, 205)),
        ]

        self.action_buttons = []
        for label, shortcut, callback, accent in action_defs:
            self.action_buttons.append(
                UIButton(
                    rect=pygame.Rect(0, 0, act_col_w, 26),
                    label=label,
                    shortcut=shortcut,
                    callback=callback,
                    accent_color=accent,
                )
            )

        simulation_defs = [
            ("Play", v.simulation_play, (65, 210, 145)),
            ("Pause", v.simulation_pause, (160, 185, 220)),
            ("Step Back", v.simulation_step_back, (120, 185, 245)),
            ("Step Forward", v.simulation_step_forward, (120, 185, 245)),
            ("Reset Time", v.simulation_reset, (245, 170, 115)),
        ]
        self.simulation_buttons = [
            UIButton(
                rect=pygame.Rect(0, 0, act_col_w, 26),
                label=label,
                callback=callback,
                accent_color=accent,
            )
            for label, callback, accent in simulation_defs
        ]

        self._relayout()

    def _relayout(self):
        """Calculates precise Y coordinates for all widgets."""
        x = self.panel_rect.x + 10
        w = self.panel_rect.width - 20
        col_w = (w - 6) // 2

        if self.is_collapsed:
            self.panel_rect.height = 42
            self.collapse_btn_rect.top = 11
            return

        y = self.panel_rect.y + 70

        # Section 1: Live Key Stats
        y += 18  # Section header space
        for stepper in self.steppers:
            stepper.rect.x = x
            stepper.rect.y = y
            stepper.rect.width = w
            btn_w = 22
            btn_h = 20
            mid_y = stepper.rect.centery
            stepper.btn_dec_rect = pygame.Rect(stepper.rect.right - (btn_w * 2 + 74), mid_y - btn_h // 2, btn_w, btn_h)
            stepper.val_rect = pygame.Rect(stepper.btn_dec_rect.right + 2, mid_y - btn_h // 2, 68, btn_h)
            stepper.btn_inc_rect = pygame.Rect(stepper.val_rect.right + 2, mid_y - btn_h // 2, btn_w, btn_h)
            y += 31

        # Simulation controls and fixed-step playback status.
        y += 4
        y += 18
        for idx, btn in enumerate(self.simulation_buttons):
            row = idx // 2
            col = idx % 2
            btn.rect.x = x + col * (col_w + 6)
            btn.rect.y = y + row * 29
            btn.rect.width = col_w
        y += ((len(self.simulation_buttons) + 1) // 2) * 29 + 4

        # Section 2: Display Toggles (2 columns)
        y += 12
        y += 18  # Section header space
        for idx, toggle in enumerate(self.toggles):
            row = idx // 2
            col = idx % 2
            toggle.rect.x = x + col * (col_w + 6)
            toggle.rect.y = y + row * 29
            toggle.rect.width = col_w

        y += ((len(self.toggles) + 1) // 2) * 29 + 8

        # Section 3: Simulation Actions (2 columns)
        y += 18  # Section header space
        for idx, btn in enumerate(self.action_buttons):
            row = idx // 2
            col = idx % 2
            btn.rect.x = x + col * (col_w + 6)
            btn.rect.y = y + row * 29
            btn.rect.width = col_w

        y += ((len(self.action_buttons) + 1) // 2) * 29 + 12

        self.panel_rect.height = y - self.panel_rect.y
        self.collapse_btn_rect.top = self.panel_rect.y + 11

        # Legend repositioning
        self.legend_rect.x = max(16, self.viewer.width - self.legend_rect.width - 16)
        self.legend_rect.y = self.viewer.height - self.legend_rect.height - 5

    def update_screen_size(self, width: int, height: int):
        """Updates UI positions when window is resized."""
        self.legend_rect.x = max(16, width - self.legend_rect.width - 16)
        self.legend_rect.y = height - self.legend_rect.height - 5
        self._relayout()

    def toggle_collapse(self):
        """Toggles between expanded HUD and minimalist header."""
        self.is_collapsed = not self.is_collapsed
        self._relayout()

    def handle_mouse_motion(self, pos: tuple[int, int]) -> bool:
        """Handles mouse hover across all UI components."""
        self.hover_collapse = self.collapse_btn_rect.collidepoint(pos)

        consumed = self.panel_rect.collidepoint(pos) or (
            self.viewer.show_boundaries and self.legend_rect.collidepoint(pos)
        )

        if not self.is_collapsed:
            for s in self.steppers:
                s.handle_mouse_motion(pos)
            for t in self.toggles:
                t.handle_mouse_motion(pos)
            for a in self.action_buttons:
                a.handle_mouse_motion(pos)
            for button in self.simulation_buttons:
                button.is_hovered = button.rect.collidepoint(pos)

        return consumed

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Handles mouse clicks on buttons, steppers, or toggles."""
        if self.collapse_btn_rect.collidepoint(pos):
            self.toggle_collapse()
            return True

        if self.is_collapsed:
            # If clicked inside collapsed title bar, expand it
            if self.panel_rect.collidepoint(pos):
                self.toggle_collapse()
                return True
            return False

        if not self.panel_rect.collidepoint(pos):
            # Check legend click (don't drag camera if clicked inside legend)
            if self.viewer.show_boundaries and self.legend_rect.collidepoint(pos):
                return True
            return False

        # Check steppers
        for stepper in self.steppers:
            if stepper.handle_click(pos):
                return True

        # Check toggles
        for toggle in self.toggles:
            if toggle.handle_click(pos):
                return True

        # Check actions
        for act in self.action_buttons:
            if act.handle_click(pos):
                return True

        for button in self.simulation_buttons:
            if button.handle_click(pos):
                return True

        # If inside panel but no specific button was hit, consume event so globe doesn't spin
        return True

    def draw(self, screen: pygame.Surface):
        """Draws the entire HUD interface with translucent backdrop and crisp styling."""
        v = self.viewer

        # 1. Main Panel Backdrop
        panel_surf = pygame.Surface((self.panel_rect.width, self.panel_rect.height), pygame.SRCALPHA)
        panel_surf.fill((14, 18, 28, 235))
        screen.blit(panel_surf, self.panel_rect.topleft)
        pygame.draw.rect(screen, (40, 56, 80), self.panel_rect, width=1, border_radius=8)

        # 2. Header Bar
        title_surf = self.title_font.render("WORLD GEN 3D", True, (245, 250, 255))
        screen.blit(title_surf, (self.panel_rect.x + 12, self.panel_rect.y + 11))

        # Collapse Button
        col_bg = (48, 66, 92) if self.hover_collapse else (28, 38, 54)
        pygame.draw.rect(screen, col_bg, self.collapse_btn_rect, border_radius=4)
        pygame.draw.rect(screen, (60, 85, 120), self.collapse_btn_rect, width=1, border_radius=4)
        sym = "+" if self.is_collapsed else "-"
        sym_surf = self.header_font.render(sym, True, (220, 230, 245))
        screen.blit(sym_surf, (self.collapse_btn_rect.centerx - sym_surf.get_width() // 2, self.collapse_btn_rect.centery - sym_surf.get_height() // 2 - 1))

        if self.is_collapsed:
            col_hint = self.small_font.render("[H] Expand Controls", True, (140, 165, 195))
            screen.blit(col_hint, (self.panel_rect.x + 140, self.panel_rect.y + 14))
            return

        # 3. Telemetry / Status Pill Row
        fps = int(v.clock.get_fps())
        tot_p = len(v.plate_generator.plates) if v.plate_generator else 0
        micro_p = sum(1 for p in v.plate_generator.plates if p.is_microplate) if v.plate_generator else 0
        oceanic_p = sum(1 for p in v.plate_generator.plates if p.is_oceanic) if v.plate_generator else 0
        continental_p = tot_p - oceanic_p
        max_micro = v.plate_generator.max_microplates if v.plate_generator else 6
        subdiv_info = f"LOD {v.render_subdiv}" if v.use_adaptive_lod else f"Subdiv {v.mesh.subdivisions}"

        tel_y = self.panel_rect.y + 36
        tel_text = f"{fps} FPS  •  {tot_p} Plates ({oceanic_p} oceanic / {continental_p} continental; {micro_p}/{max_micro} μ)  •  {subdiv_info}"
        tel_surf = self.small_font.render(tel_text, True, (135, 160, 190))
        screen.blit(tel_surf, (self.panel_rect.x + 12, tel_y))

        # Thin separator line
        pygame.draw.line(screen, (32, 44, 64), (self.panel_rect.x + 10, tel_y + 19), (self.panel_rect.right - 10, tel_y + 19), 1)

        # 4. Section 1 Header: KEY STATS
        s1_title = self.section_font.render("LIVE KEY PARAMETERS", True, (110, 175, 235))
        screen.blit(s1_title, (self.panel_rect.x + 12, self.panel_rect.y + 66))

        for stepper in self.steppers:
            stepper.draw(screen, self.font, self.small_font)

        # Section 2 Header: DISPLAY TOGGLES
        s2_y = self.toggles[0].rect.y - 17
        s2_title = self.section_font.render("DISPLAY TOGGLES", True, (110, 175, 235))
        screen.blit(s2_title, (self.panel_rect.x + 12, s2_y))

        for toggle in self.toggles:
            toggle.draw(screen, self.font, self.small_font)

        sim_y = self.simulation_buttons[0].rect.y - 17
        simulation = v.simulation
        elapsed = simulation.elapsed_myr if simulation else 0.0
        playback = "PLAYING" if simulation and simulation.is_playing else "PAUSED"
        sim_title = self.section_font.render(
            f"SIMULATION · {elapsed:.1f} Myr · {playback}", True, (110, 205, 165)
        )
        screen.blit(sim_title, (self.panel_rect.x + 12, sim_y))
        for button in self.simulation_buttons:
            button.draw(screen, self.font, self.small_font)

        # Section 3 Header: ACTIONS
        s3_y = self.action_buttons[0].rect.y - 17
        s3_title = self.section_font.render("ACTIONS & RESETS", True, (110, 175, 235))
        screen.blit(s3_title, (self.panel_rect.x + 12, s3_y))

        for btn in self.action_buttons:
            btn.draw(screen, self.font, self.small_font)

        # 5. Boundary Legend (Bottom Card)
        if v.show_boundaries:
            leg_surf = pygame.Surface((self.legend_rect.width, self.legend_rect.height), pygame.SRCALPHA)
            leg_surf.fill((14, 18, 28, 225))
            screen.blit(leg_surf, self.legend_rect.topleft)
            pygame.draw.rect(screen, (40, 56, 80), self.legend_rect, width=1, border_radius=6)

            leg_title = self.section_font.render("PLATE COLORS & BOUNDARY STRESS", True, (180, 195, 215))
            screen.blit(leg_title, (self.legend_rect.x + 10, self.legend_rect.y + 8))

            legend_items = [
                ((75, 115, 175), "Oceanic density: teal low → purple high"),
                ((195, 95, 55), "Continental density: light orange low → deep red high"),
                ((165, 165, 165), "Lithosphere height: darker low → brighter high"),
                ((255, 60, 60), "Convergent (Compression / Collision)"),
                ((60, 180, 255), "Divergent (Rifting / Spreading)"),
                ((255, 220, 50), "Transform (Lateral Strike-Slip)"),
            ]

            for idx, (color, text) in enumerate(legend_items):
                ly = self.legend_rect.y + 24 + idx * 17
                pygame.draw.rect(screen, color, (self.legend_rect.x + 10, ly + 2, 12, 8), border_radius=2)
                txt_surf = self.small_font.render(text, True, (215, 225, 235))
                screen.blit(txt_surf, (self.legend_rect.x + 28, ly))
